import pandas as pd
import numpy as np
import logging

import statsmodels.api as sm
from typing import Tuple, Any

logger = logging.getLogger(__name__)
EPS = 1.0e-6  # Error margin
# QUAL_NAME_LIST = ["cu", "conc", "as", "mo", "fe", "zn", "rec", "pb", "spi"]
OLD_STOCK_PCT = 0.1  # % of the pile used to estimate output proportions (mix)
OLD_STOCK_MIN = 4000  # max old tonnage of pile to be used for output proportions
MIN_TON_CBELT = 200  # min avg tonnage in cbelt to be considered in mass balance
MAX_TOL_BALANCE = 0.3
QUAL_NAME_LIST = ["cu", "mo"]
SOURCE_NAME_LIST = ["rajo", "subte", "rt"]


def replace_nan_by_mean(df: pd.DataFrame, tag_list: list) -> pd.DataFrame:
    """For each tag in list, replace its missing values by the average of the other columns."""

    data = df.copy()
    data[tag_list] = data[tag_list].apply(
        lambda x: x.fillna(x[tag_list].mean()), axis=1
    )

    if data[tag_list].isna().sum().sum() != 0:
        raise ValueError("Found missing values for columns in ´tag_list´")

    return data


def repair_missing_qual(
    data: pd.DataFrame,
) -> pd.DataFrame:  # TODO: improve this function
    """Replace missing qualities by their average.

    Args:
        data: Dataset to be corrected for missing values.

    Returns:
        df: Corrected dataset.

    """

    df = data.copy()
    dispatch_name = "tonelaje_dispatch"
    for qual in QUAL_NAME_LIST:
        for source in SOURCE_NAME_LIST:
            qual_col = f"{source}_{qual}"
            ton_col = f"{source}_{dispatch_name}"
            # First, select samples where ton > 0 and impute by the average
            mask = ~pd.isnull(df[ton_col])
            # print(mask)
            df_masked = df.loc[mask, qual_col].copy()
            qual_mean = df_masked.mean()
            n_missing = df_masked.isna().sum()
            # Count and replace missing by mean
            if n_missing > 0:
                pct_missing = n_missing / df_masked.shape[0]
                if pct_missing > 0.5:  # Min valid values in qualities
                    raise ValueError(
                        f"{100 * pct_missing:.1f}% missing entries for ´{qual}´ in source ´{source}´"
                    )
                logger.warning(
                    f"{n_missing} missing entries for ´{qual}´ in source ´{source}´ were replaced "
                    f"({100 * pct_missing:.1f}%)"
                )
                df.loc[mask, qual_col] = df.loc[mask, qual_col].fillna(qual_mean)
            # Second, impute all missing by mean as well
            df[qual_col] = df[qual_col].fillna(qual_mean)
            # Validate no missing data remains
            if df[qual_col].isna().sum() != 0:
                raise ValueError(
                    f"Missing entries for ´{qual}´ in source ´{source}´ were found"
                )
    return df


def _get_negligible_cbelts(df: pd.DataFrame, cbelt_list: list) -> list:
    """Remove cbelts from list if avg tonnage is below threshold.

    Args:
        df: Dataset containing values for tags.
        cbelt_list: List of ConveyorBelt objects whose data will be compared against
        threshold value.

    Returns:
        cbelts_to_exclude: List of cbelts whose avg tonnage is below threshold.

    """

    cbelts_to_exclude = []
    for cbelt in cbelt_list:
        if df[cbelt.tag].mean() < MIN_TON_CBELT:
            cbelts_to_exclude.append(cbelt)
            logger.warning(
                f"{cbelt.__class__.__name__} {cbelt.name} excluded from mass balance due to "
                f"insufficient input mass."
            )

    return cbelts_to_exclude


def find_stock_level_factor(df: pd.DataFrame, tag_level: str, tag_stock: str) -> float:
    """Calculate calibration factor to fit stock level to stock in tonnes.
    # TODO: improve this for cases where level sensor has a lower limit
    Regularized linear regression is used to find the coefficient.

    Args:
        df: Dataset from which variables will be extracted.
        tag_level: PI tag containing stock level (usually a % value).
        tag_stock: PI tag containing stock in tonnes.

    Returns: Regression coefficient.
    """
    df_clean = df[[tag_level, tag_stock]].dropna()

    x = df_clean[tag_level]
    y = df_clean[tag_stock]

    ols = sm.OLS(y, x)
    ols_reg = ols.fit_regularized(
        method="elastic_net",
        alpha=0.0,
        L1_wt=1.0,
        start_params=None,
        profile_scale=False,
        refit=False,
    )
    # Check for convergence
    if not ols_reg.converged:
        raise ValueError("Unable to find a calibration factor for stock")
    # Return regression factor
    return ols_reg.params[tag_level]


def _invert_dict_sign(d: dict):
    """Invert signs of values in input dictionary"""
    return {key: -value for key, value in d.items()}


def _update_stock(master_dict: dict, new_dict: dict) -> dict:
    """Add values from one dict to the other, plus additional checks.

    It is assumed that master_dict contains all keys from new_dict,
    possibly in addition to 'stock_total'.

    Args:
        master_dict: Initial dictionary to be modified.
        new_dict: Dictionary of values to be added to master_dict.

    Returns:
        master_dict: Updated master dictionary.

    """
    # Make sure stock is not negative (up to rounding error)
    try:
        if min(master_dict.values()) < -EPS:
            raise ValueError(
                f"Stock dictionary {master_dict} has a negative value which is not allowed"
            )
    except TypeError:
        print(master_dict, new_dict)

    # Add values key by key
    for key in new_dict.keys():
        updated_value = np.nansum([master_dict[key], new_dict[key]])
        if updated_value < -EPS:
            raise ValueError(
                f"Stock dictionary {master_dict} would encounter a negative value for key ´{key}´ "
                f"if added to {new_dict}. This is not allowed"
            )
        master_dict[key] = updated_value
        # Ensure result is non-negative and significant

        # If slightly negative, replace by 0
        master_dict[key] = max(0, master_dict[key])
        # Round down if small value
        if master_dict[key] < EPS:
            master_dict[key] = 0
        # Update stock_total if key exists
        if "stock_total" in master_dict.keys():
            master_dict["stock_total"] = max(
                0, np.nansum([master_dict["stock_total"], new_dict[key]])
            )
    return master_dict


def _update_cum_qualities(cum_qual: dict, new_qual: dict, stock_in: dict) -> dict:
    """Add cumulative qualities (e.g. Cu [kg]) using total incoming stock as sum weight.

    This function calculates the 'numerator' of a weighted average to estimate total quality (e.g. total Cu [kg]).
    This output must be normalized later on. It is assumed that master_dict contains all keys from new_dict.

    Args:
        cum_qual: Cumulative qualities (e.g. Cu [kg])
        new_qual: New qualities as a unit of mass (e.g. Cu%)
        stock_in: Amount of stock (mass) being added.

    Returns:
        qual_in: Updated cumulative qualities.
    """
    # Make sure quality is not negative (up to rounding error)
    if min(cum_qual.values()) < 0 or min(new_qual.values()) < 0:
        raise ValueError("Qualities cannot be negative")
    # Calculate total stock coming in
    total_stock_in = sum(stock_in.values())
    # Add cum qualities from new stock
    for qual in new_qual.keys():
        cum_qual[qual] = np.nansum([cum_qual[qual], new_qual[qual] * total_stock_in])

    return cum_qual


def _update_current_qualities(qual_profile: dict, stock_profile: dict) -> dict:
    """Update aggregated qualities based on complete profile.

    Used to calculate overall qualities of a stock pile, given its granular profile. Uses total
    stock entered in each timestamp as the weight of the sum.

    Args:
        qual_profile: Dictionary indexed by timestamps, with keys for each quality being monitored.
        stock_profile: Dictionary indexed by timestamps, with key 'stock_total' containing total material.
    """
    cum_qual = {qual: 0 for qual in QUAL_NAME_LIST}  # Saves total quality in the pile
    stock_total = 0  # Saves total stock in pile
    # Calculate total stock and total qualities in pile
    for ts in qual_profile:
        stock_total += stock_profile[ts]["stock_total"]
        for qual in QUAL_NAME_LIST:
            cum_qual[qual] += qual_profile[ts][qual] * stock_profile[ts]["stock_total"]
    # Normalize qualities over the entire pile
    return _normalize_qualities(cum_qual, stock_total)


def _normalize_qualities(cum_qual: dict, stock_total: float) -> dict:
    """Divide values of input dictionary by stock_total."""
    if stock_total == 0:
        return cum_qual
    else:
        return {qual: val / stock_total for qual, val in cum_qual.items()}


def is_dict_positive(material_dict: dict) -> bool:
    """Check that all values in dict are positive."""
    return np.all([val >= 0 for val in material_dict.values()])


def is_profile_positive(profile_dict: dict) -> bool:
    """Check that all dictionaries in profile are positive. """
    return np.all(
        [is_dict_positive(material_dict) for material_dict in profile_dict.values()]
    )


class ConveyorBelt:
    """A class used to represent a conveyor belt.

    The main functionality is to store the ore composition and qualities after running a simulation of
    a connecting ConveyorBelt.

    Main methods:
    update_history: Updates ore composition and qualities of the material being transported (charged
    from or discharged to) a ConveyorBelt object.
    stock_history_to_frame: Generate a Pandas DataFrame with the ore composition over time after
    a simulation is run.
    quality_history_to_frame: Generate a Pandas DataFrame with the ore qualities over time after
    a simulation is run.

    Main attributes:
    stock_history: Time-indexed history, current stock for full simulation, e.g. {'20/10/04': current_stock}
    qual_history: Time-indexed history, current qualities for full simulation, e.g. {'20/10/04': current_qual}
    balance_factor_history: Time-index history, balance factor calculated over time after a simulation is run.

    """

    def __init__(
        self,
        name: str,
        max_capacity: int,
        tag: str,
        source: str = None,
        in_cbelt: Any = None,
    ) -> None:
        """Create instance and initialize attributes.

        Args:
            name: Human-readable name for the ConveyorBelt.
            max_capacity: Max capacity in tonnes.
            tag: PI tag containing measured tonnage.
            source: (Optional) One value from SOURCE_NAME_LIST
            in_cbelt: (Optional) Name of incoming cbelt from which qualities and stock are brought in

        """
        self.name = name
        self.max_capacity = max_capacity  # TODO: add warning if exceeded
        self.tag = tag
        self.source = source
        self.qual_names = QUAL_NAME_LIST
        self.in_cbelt = in_cbelt
        self.stock_history = {}
        self.qual_history = {}
        self.balance_factor_history = {}

    def update_history(
        self, row: pd.Series, source_pct: dict = None, qual_dict: dict = None
    ) -> None:
        """Update stock and qualities history for a belt.

        Assumptions:
        - source_pct and qual_dict contain sources as dictionary keys
        - 'source_pct' represents the proportion of each source in the total
        - df contains columns named 'source_qual' for all sources and qualities

        Args:
            row: Row from a dataframe containing one time sample from master data table, indexed by timestamp.
            source_pct: Dictionary of values adding up to 1.
            qual_dict: Dictionary of qualities for each source.
        """
        # Get date
        date = row.name
        # Case: CBelt has known source material, so automatically add history from df
        if self.source is not None:
            self.stock_history[date] = {self.source: row[self.tag]}
            self.qual_history[date] = {
                qual: row[f"{self.source}_{qual}"] for qual in self.qual_names
            }
        # Case: CBelt transfers mixed material, so it needs to be updated from a stock pile using function arguments
        elif (source_pct is not None) & (qual_dict is not None):
            # Calculate tonnage for each source based on provided values for source_pct
            self.stock_history[date] = {
                source: row[self.tag] * source_pct[source] for source in source_pct
            }
            self.qual_history[date] = qual_dict.copy()
        # Case: CBelt is a continuation of another upstream CBelt, so we bring info proportional to the tonnages
        elif self.in_cbelt is not None:
            stock_in = sum(self.in_cbelt.stock_history[date].values())
            # Calculate ratio between current and input cbelt tonnages
            if stock_in != 0:
                ratio = row[self.tag] / stock_in
            else:
                ratio = 0
            if ratio > 1 + EPS:  # Make sure no mass is being "created"
                raise ValueError(
                    f"{self.__class__.__name__} {self.name} has more tonnage than input cbelt"
                )
            # Calculate source distribution based on input cbelt and ratio
            self.stock_history[date] = {
                source: val * ratio
                for source, val in self.in_cbelt.stock_history[date].items()
            }
            # Bring qualities from input cbelt
            self.qual_history[date] = self.in_cbelt.qual_history[date].copy()
        # Any other case, error in arguments of the function
        else:
            raise ValueError("Wrong argument combination for this function")

        # Check that stock history does not have missing data
        if any(np.isnan(val) for val in self.stock_history[date].values()):
            print(self.stock_history[date])
            print(source_pct, qual_dict, self.name)
            raise ValueError(f"NaN value encountered in stock_history on {date}")
        # Check that, if ton > 0, qual history does not have missing data
        if (row[self.tag] != 0) & (
            any(np.isnan(val) for val in self.qual_history[date].values())
        ):
            raise ValueError(f"NaN value encountered in qual_history on {date}")

    def get_stock_history(self, df: pd.Series) -> dict:
        """Get stock history for the specified timestamp."""
        return self.stock_history[df.name]

    def get_qual_history(self, df: pd.Series) -> dict:
        """Get qualities history for the specified timestamp."""
        return self.qual_history[df.name]

    def stock_history_to_frame(self):
        """Get complete stock history as a Pandas DataFrame"""
        return pd.DataFrame.from_dict(self.stock_history, orient="index").sort_index()

    def qual_history_to_frame(self):
        """Get complete qualities history as a Pandas DataFrame"""
        return pd.DataFrame.from_dict(self.qual_history, orient="index").sort_index()

    def balance_factor_history_to_series(self):
        return pd.Series(self.balance_factor_history).rename(self.name)


class StockPile:
    """A class used to represent a Stock Pile.

    The main functionality is to run a simulation of the time-evolution of ore composition and qualities
    as ConveyorBelts change and discharge material.

    Main methods:
        run: Executes the simulation over the provided dataset. It can be run for consecutive periods.
        stock_history_to_frame: Generate a Pandas DataFrame with the ore composition over time after
        the simulation is run.
        quality_history_to_frame: Generate a Pandas DataFrame with the ore qualities over time after
        the simulation is run.
        balance_upstream_mass: Correct input dataset to enforce mass balance across input/output/stock
        introducing balance factors to selected input cbelts.

    Main attributes:
        current_stock: Current stock distribution, e.g. {'stock_total: 7, 'mina': 4, 'rt': 3}
        stock_history: Time-indexed history, current stock for full simulation, e.g. {'20/10/04': current_stock}
        stock_profile: Time-indexed profile, stock data for all material currently in the stock
        residence_time: Time-indexed history of residence time [h] for full simulation
        current_qual: Current qualities, e.g. {'cu': 0.5, 'mo': 0.1}
        qual_history: Time-indexed history, current qualities for full simulation, e.g. {'20/10/04': current_qual}
        qual_profile: Time-indexed profile, qualities data for all material currently in the stock

    """

    def __init__(
        self,
        name: str,
        max_capacity: int,
        in_cbelts: Tuple[ConveyorBelt, ...],
        out_cbelts: Tuple[ConveyorBelt, ...],
        stock_level_tag: str,
        stock_ton_tag: str,
    ) -> None:
        """Create instance and initialize attributes.

        Args:
            name: Human-readable name for the StockPile.
            max_capacity: Max capacity in tonnes.
            in_cbelts: List of ConveyorBelt objects that feed the StockPile.
            out_cbelts: List of ConveyorBelt objects that exit from the StockPile.
            stock_level_tag: PI tag containing the stock level (%) for this pile.
            stock_ton_tag: PI tag containing the stock (ton) for this pile.

        """
        self.name = name
        self.max_capacity = max_capacity
        self.source_names = SOURCE_NAME_LIST
        self.qual_names = QUAL_NAME_LIST
        self.in_cbelts = in_cbelts
        self.out_cbelts = out_cbelts
        self.stock_level_tag = stock_level_tag
        self.stock_ton_tag = stock_ton_tag

        # Stock
        self.stock_current = {"stock_total": None}
        self.stock_current.update({source: None for source in self.source_names})
        self.stock_history = {}
        self.stock_profile = {}
        self.residence_time = {}

        # Qualities
        self.qual_current = {qual: None for qual in self.qual_names}
        self.qual_history = {}
        self.qual_profile = {}

        # Constant parameters
        self._OLD_STOCK_PCT = OLD_STOCK_PCT

    def set_init_inventory(
        self,
        data: pd.DataFrame,
        ton_by_source: dict,
        qualities: dict,
        ton_inventory: int = None,
    ) -> None:
        """Set initial stock and qualities for a new StockPile object.

        Args:
            data: Dataset to be used in the simulation, sorted by date.
            ton_by_source: Proportion (as % or [ton]) of each source in the starting stock pile.
            qualities: Starting qualities for the stock pile.
            ton_inventory: (Optional) Initial tonnes in the stock pile.

        Assumptions:
        - First row of data represents the starting point of the simulation
        - If ton_inventory is not provided, it will be estimated from the input data

        """
        # Make sure StockPile is new
        if self.stock_current["stock_total"] is not None:
            logger.warning(f"{self.__class__.__name__} has been re-initialized.")
        # Case: initial inventory tonnage is not provided, so it's obtained from input data
        if ton_inventory is None:
            # Find calibration factor between stock level and tonnage
            stock_level_factor = find_stock_level_factor(
                data, self.stock_level_tag, self.stock_ton_tag
            )
            # Calculate starting inventory
            ton_inventory = stock_level_factor * data[self.stock_level_tag][0]

        # Set initial total stock
        self.stock_current["stock_total"] = ton_inventory

        # Set ton by source (either as % from total or ton)
        if sum(ton_by_source.values()) == 1:
            for source in self.source_names:
                self.stock_current[source] = (
                    self.stock_current["stock_total"] * ton_by_source[source]
                )
        elif sum(ton_by_source.values()) == self.stock_current["stock_total"]:
            for source in self.source_names:
                self.stock_current[source] = ton_by_source[source]
        else:
            raise ValueError("Wrong argument for 'ton_by_source'")

        # Set qualities
        self.qual_current = qualities

    def restart(self) -> None:
        """Reset all stock and qualities attributes."""
        # Stock
        self.stock_current = dict.fromkeys(self.stock_current, None)
        self.stock_profile = {}
        self.stock_history = {}
        self.residence_time = {}
        # Qualities
        self.qual_current = dict.fromkeys(self.qual_current, None)
        self.qual_profile = {}
        self.qual_history = {}
        # Restart cbelts attached to pile
        for cbelt in self.in_cbelts + self.out_cbelts:
            cbelt.stock_history = {}
            cbelt.qual_history = {}
            cbelt.balance_factor_history = {}

    def run(self, df: pd.DataFrame):
        """Run the simulation.

        The provided dataset will be used to run the traceability simulation over the time range.
        This method accepts consecutive runs, which can be used to split the simulation into
        separate time periods.

        Args:
            df: time-indexed dataset containing all required data to carry out the simulation.

        """
        # Check validity of index in df
        self._check_index_df(df)
        # Initialize stock
        self._set_init_state(df)
        # Iterate over time samples
        for ts, row in df.iterrows():
            # Add input cbelts
            mass_in = self._input_cbelts(row)
            # Subtract output cbelts
            mass_out = self._output_cbelts(row)
            # Save history and residence time
            self.stock_history[ts] = self.stock_current.copy()
            self.qual_history[ts] = self.qual_current.copy()
            self.residence_time[ts] = self._calculate_residence_time()
            # Check mass balance
            self._check_mass_balance(mass_in, mass_out)

    def _input_cbelts(self, row: pd.Series) -> dict:
        """Update stock and qualities based on incoming material.

        Args:
            row: Data sample indexed by a timestamp.
        Returns:
            ton_in: incoming material indexed by source.

        """
        # Get date to index results
        date = row.name
        # Calculate how much material is entering across input cbelts
        ton_in = {source: 0 for source in self.source_names}
        cum_qual = {qual: 0 for qual in self.qual_names}
        for cbelt in self.in_cbelts:
            # Get stock and qualities from cbelt
            try:
                cbelt.update_history(row=row)  # Update cbelt with incoming material
            except ValueError as e:
                print(e)
                print("AQUI 1")
                print(date, cbelt.name, ton_in)
            new_stock = cbelt.get_stock_history(row)
            new_quals = cbelt.get_qual_history(row)
            # Update incoming stock and qualities
            try:
                ton_in = _update_stock(ton_in, new_stock)
            except ValueError as e:
                print(e)
                print("AQUI 2")
                print(date, cbelt.name, ton_in, new_stock)
            cum_qual = _update_cum_qualities(cum_qual, new_quals, new_stock)
        # Normalize qualities
        qual_in = _normalize_qualities(cum_qual, sum(ton_in.values()))
        # Update profiles (by hour)
        self.stock_profile[date] = ton_in.copy()
        self.stock_profile[date]["stock_total"] = sum(ton_in.values())
        self.qual_profile[date] = qual_in.copy()
        # Update current stock and qualities
        try:
            self.stock_current = _update_stock(self.stock_current, ton_in)
        except ValueError as e:
            print(e)
            print("AQUI2.5")
            print(date, self.stock_current, ton_in)
        self.qual_current = _update_current_qualities(
            self.qual_profile,
            self.stock_profile,
        )
        return ton_in

    def _output_cbelts(self, row: pd.Series) -> dict:
        """Update stock and qualities based on outgoing material.

        Args:
            row: Data sample indexed by a timestamp.
        Returns:
            ton_out_adjusted: outgoing material indexed by source.

        """
        # Calculate material blend to be discharged
        source_pct, qual_out = self._get_source_features()
        # Calculate how much material must exit
        ton_out = {source: 0 for source in self.source_names}
        for cbelt in self.out_cbelts:
            # Update cbelt with outgoing material
            cbelt.update_history(row=row, source_pct=source_pct, qual_dict=qual_out)
            # Update outgoing stock
            try:
                ton_out = _update_stock(ton_out, cbelt.get_stock_history(row))
            except ValueError as e:
                print("AQUI3")
                print(e)
                print(ton_out, cbelt.get_stock_history(row))
        # Adjust ton_out in case a source is exhausted
        ton_out_adjusted = self._adjust_ton_out(ton_out)
        # Update stock_profile
        self.stock_profile = self._update_stock_profile(ton_out_adjusted)
        # Update qualities profile
        self.qual_profile = self._update_qualities_profile()
        # Update current stock and qualities
        try:
            self.stock_current = _update_stock(
                self.stock_current, _invert_dict_sign(ton_out_adjusted)
            )
        except ValueError as e:
            print("AQUI4")
            print(e)
            print(self.stock_current, _invert_dict_sign(ton_out_adjusted))
            print(self.stock_profile_to_frame().tail())
        self.qual_current = _update_current_qualities(
            self.qual_profile,
            self.stock_profile,
        )
        # Ensure values are positive
        if not is_dict_positive(self.stock_current):
            raise ValueError("Current stock dict has negative entries")
        if not is_dict_positive(self.qual_current):
            raise ValueError("Current qualities dict has negative entries")
        if not is_profile_positive(self.stock_profile):
            raise ValueError("Stock profile has negative entries")
        if not is_profile_positive(self.qual_profile):
            raise ValueError("Qualities profile has negative entries")
        # Return adjusted outgoing material
        return ton_out_adjusted

    def _adjust_ton_out(self, ton_out: dict) -> dict:
        """Adjust outgoing material based on stock availability.

        This is needed to tackle situations where outgoing cbelt requires more
        material from a source than currently available at the pile. In this case,
        tonnage is borrowed from the current predominant source.

        Args:
            ton_out: Dictionary of materials indexed by source.

        Returns:
            ton_out_adjusted: Adjusted dictionary of materials.

        """
        # Calculate total existing stock by source and sort by stock tonnage
        stock_sum = self.stock_profile_to_frame()[self.source_names].sum(axis=0)
        sorted_sources = stock_sum.sort_values(ascending=False).index.to_list()
        # Make sure the pile has enough material
        if stock_sum.sum() < sum([val for (key, val) in ton_out.items()]):
            raise ValueError(
                f"Current stock of the {self.__class__.__name__} '{self.name}' is {stock_sum.to_dict()}"
                f" whereas ore being pulled is {ton_out} which exceeds existing tonnage"
            )
        # Update ton_out based on material availability for each source
        ton_out_adjusted = ton_out.copy()
        residue = 0  # How much excess material will be supplied by other sources
        for source in sorted_sources:
            if ton_out[source] > stock_sum[source]:
                residue += ton_out[source] - stock_sum[source]
                ton_out_adjusted[source] = stock_sum[source]
        # Find sources that have leftover stock
        sources_with_stock_left = {
            source: stock_sum[source] - ton_out_adjusted[source]
            for source in sorted_sources
            if stock_sum[source] - ton_out_adjusted[source] > 0  # EPS
        }
        # Subtract residual from available sources (in decreasing order)
        for source, _ in sorted(
            sources_with_stock_left.items(), key=lambda x: x[1], reverse=True
        ):
            if sources_with_stock_left[source] >= residue:
                ton_out_adjusted[source] += residue
                residue = 0
                break
            else:
                ton_out_adjusted[source] += sources_with_stock_left[source]
                residue -= sources_with_stock_left[source]
        else:
            raise ValueError("Residual material cannot be supplied by other sources")
        # Sanity checks
        if residue > EPS:
            raise ValueError("Residual material was not supplied by other sources")
        if not is_dict_positive(ton_out_adjusted):
            raise ValueError("Outgoing material contains a negative value")
        if np.abs(sum(ton_out.values()) - sum(ton_out_adjusted.values())) > EPS:
            raise ValueError(
                "Total adjusted outgoing material differs from actual tonnage"
            )
        # Return adjusted tonnage
        return ton_out_adjusted

    def _set_init_state(self, df: pd.DataFrame) -> None:
        """Initialize stock and qualities history and profile.

        Args:
            df: Dataset to be used in the simulation.

        """
        # Define t0 as 1 hour to the past
        t0 = df.sort_index().index[0] + pd.Timedelta(hours=-1)
        # Case: New StockPile
        if len(self.stock_history) == 0:
            # If StockPile has not been initialized
            if self.stock_current["stock_total"] is None:
                raise ValueError(
                    f"{self.__class__.__name__} {self.name} has not been initialized. "
                    f"Run method ´set_init_inventory´ first."
                )
            # Stock
            self.stock_history[t0] = self.stock_current.copy()
            self.stock_profile[t0] = self.stock_current.copy()
            # Qualities
            self.qual_history[t0] = self.qual_current.copy()
            self.qual_profile[t0] = self.qual_current.copy()
        # Case: Existing StockPile with time gap (deleted period)
        elif t0 not in self.stock_history.keys():
            # Stock
            self.stock_history[t0] = self.stock_current.copy()  # Add to history
            self.stock_profile = {t0: self.stock_current.copy()}  # Restart profile
            # Qualities
            self.qual_history[t0] = self.qual_current.copy()  # Add to history
            self.qual_profile = {t0: self.qual_current.copy()}  # Restart profile
        # Case: Existing StockPile without time gaps
        else:
            pass

    def _update_stock_profile(self, ton_out: dict) -> dict:
        """Update stock profile after outgoing material has been calculated.

        Args:
            ton_out: Tonnage to be removed from the pile for each source.

        Returns:
            stock_profile: Updated stock profile.

        """
        # Iteratively remove stock from each source until reaching ton_out
        stock_profile = self.stock_profile.copy()
        # Get list of sorted timestamps
        sorted_profile = sorted(stock_profile)
        # Subtract the outgoing material by source
        cum_sum_source = {source: 0 for source in self.source_names}
        for source in self.source_names:
            for ts in sorted_profile:
                cum_sum_source[source] += stock_profile[ts][source]
                if cum_sum_source[source] < ton_out[source]:
                    stock_profile[ts][source] = 0
                else:
                    stock_profile[ts][source] = cum_sum_source[source] - ton_out[source]
                    break
            else:
                # ton_out should not demand more than the existing stock (check it's not rounding error)
                if np.abs(cum_sum_source[source] - ton_out[source]) > EPS:
                    raise ValueError(
                        f"Outgoing material depleted existing stock for {source}. Current stock "
                        f"is {self.stock_current} and material pulled is {ton_out}"
                    )
        # Update stock_total
        for ts in stock_profile.keys():
            stock_profile[ts]["stock_total"] = sum(
                [stock_profile[ts][key] for key in self.source_names]
            )
        # Remove empty entries (material is gone)
        for ts in sorted_profile:
            if stock_profile[ts]["stock_total"] < EPS:
                del stock_profile[ts]
        # Check positive stock
        if not is_profile_positive(stock_profile):
            raise ValueError("Resulting stock profile contains negative entries")
        # Return stock profile
        return stock_profile

    def _get_source_features(self) -> Tuple[dict, dict]:
        """Calculate outgoing blend based on current availability in the pile.

        Determine the source distribution and qualities to be pushed to outgoing cbelts,
        using a % of the oldest material in stock as a reference to simulate local mixing.

        Returns:
            pct_dict: % of material from each source.
            qual_dict: Qualities for each source.

        """
        # Calculate old stock
        stock_total = self.stock_current["stock_total"]
        old_stock = np.min([OLD_STOCK_MIN, self._OLD_STOCK_PCT * stock_total])
        # Calculate cum sum of stock
        sorted_profile = sorted(self.stock_profile)
        cum_sum_source = {source: 0 for source in self.source_names}
        cum_qual = {qual: 0 for qual in self.qual_names}
        for ts in sorted_profile:
            for source in self.source_names:
                cum_sum_source[source] += self.stock_profile[ts][source]
            for qual in self.qual_names:
                cum_qual[qual] += (
                    self.qual_profile[ts][qual] * self.stock_profile[ts]["stock_total"]
                )
            cum_sum_stock = sum(cum_sum_source.values())
            if cum_sum_stock > old_stock:
                break
        else:
            raise ValueError("Old tonnage was not achievable")
        # Calculate source pct
        pct_dict = {}
        for source in self.source_names:
            pct_dict[source] = cum_sum_source[source] / cum_sum_stock

        if np.abs(sum(pct_dict.values()) - 1) > EPS:
            raise ValueError("Sum of source % is not equal to 1")
        # Normalize qualities
        qual_dict = _normalize_qualities(cum_qual, cum_sum_stock)
        # Return dictionaries
        return pct_dict, qual_dict

    def stock_history_to_frame(self) -> pd.DataFrame:
        return pd.DataFrame.from_dict(self.stock_history, orient="index").sort_index()

    def qual_history_to_frame(self) -> pd.DataFrame:
        return pd.DataFrame.from_dict(self.qual_history, orient="index").sort_index()

    def stock_profile_to_frame(self) -> pd.DataFrame:
        return pd.DataFrame.from_dict(self.stock_profile, orient="index").sort_index()

    def qual_profile_to_frame(self) -> pd.DataFrame:
        return pd.DataFrame.from_dict(self.qual_profile, orient="index").sort_index()

    def residence_time_to_series(self) -> pd.Series:
        return pd.Series(self.residence_time).rename(self.name)

    def _calculate_residence_time(self) -> float:
        """Calculate dynamic residence time based on current oldest material in the pile."""
        # Select material above min tonnage (to avoid small material residues in calculation)
        min_ton = 300  # TODO: to params
        filtered_profile = {
            key: val
            for (key, val) in self.stock_profile.items()
            if val["stock_total"] > min_ton
        }
        sorted_profile = sorted(filtered_profile)
        # Calculate time
        if len(sorted_profile) == 0:
            return 0
        else:
            return (
                sorted_profile[-1] - sorted_profile[0]
            ).total_seconds() / 3_600  # [h]

    def get_mean_residence_time(self) -> float:
        return np.nansum([val for _, val in self.residence_time.items()]) / len(
            self.residence_time
        )

    def _check_mass_balance(self, mass_in: dict, mass_out: dict) -> None:
        """Check that for current iteration mass is preserved for each source.

        Args:
            mass_in: Incoming material by source.
            mass_out: Outgoing material by source.

        """
        # Get latest change in stock
        diff_stock = self.stock_history_to_frame().iloc[-2:].diff().iloc[-1]

        # Check mass balance for each source
        if np.any(
            [
                diff_stock[source] - mass_in[source] + mass_out[source] > EPS
                for source in self.source_names
            ]
        ):
            raise ValueError("Mass conservation as been violated")

    def _update_qualities_profile(self) -> dict:
        """Remove entry from qualities profile if the corresponding stock entry was removed."""
        qual_profile = self.qual_profile.copy()
        for ts in self.qual_profile:
            if ts not in self.stock_profile:
                del qual_profile[ts]
        return qual_profile

    def balance_upstream_mass(
        self, df: pd.DataFrame, exclude_in_cbelt=None
    ) -> pd.DataFrame:
        """Correct cumulative mass unbalance by adjusting tonnage of upstream cbelts.

        The process determines the stock difference between first and last periods of the dataset
        and calculates a balance factor so: input + stock change = output. The balance factor is
        then applied to selected input cbelts, propagating the correction upstream.

        Args:
            df: Dataset used to balance total mass. It should match the data to be used in a
            traceability simulation later on.
            exclude_in_cbelt: List of input ConveyorBelt for which the balance factor should not
            be applied. This is used in case their tonnage is reliable and no correction is needed.

        Returns:
            data: Corrected dataset after applying balance factor to appropriate variables.

        """
        if exclude_in_cbelt is None:
            exclude_in_cbelt = []
        # Select data to be used
        data = df.copy()
        input_tags = [cbelt.tag for cbelt in self.in_cbelts]
        output_tags = [cbelt.tag for cbelt in self.out_cbelts]

        # Get initial and end stock levels for the time range
        level_start, level_end = (
            data[self.stock_level_tag][0],
            data[self.stock_level_tag][-1],
        )
        # Get conversion factor by fitting stock level and tonnage stock
        conv_factor = find_stock_level_factor(
            data, self.stock_level_tag, self.stock_ton_tag
        )
        # Calculate cumulated stock
        cum_stock = conv_factor * (level_end - level_start)
        # If the calculation failed, try using stock diff directly
        if np.isnan(cum_stock):
            cum_stock = data[self.stock_ton_tag][-1] - data[self.stock_ton_tag][0]
        if np.isnan(cum_stock):
            raise ValueError("Unable to determine cumulative stock in the period")

        # Mass balance
        output_mass = data[output_tags].sum().sum()
        exclude_in_cbelt = exclude_in_cbelt + _get_negligible_cbelts(
            df, [cbelt for cbelt in self.in_cbelts if cbelt not in exclude_in_cbelt]
        )

        input_tag_excl = [cbelt.tag for cbelt in exclude_in_cbelt]
        # Get included cbelts
        input_tag_incl = [tag for tag in input_tags if tag not in input_tag_excl]
        if len(input_tag_incl) == 0:  # No input cbelts to be balanced
            return data
        input_mass_incl = df[input_tag_incl].sum().sum()  # Mass of included cbelts
        input_mass_excl = data[input_tag_excl].sum().sum()
        balance_factor = (output_mass - input_mass_excl + cum_stock) / input_mass_incl
        # Check value of balance factor
        if np.isnan(balance_factor):
            print(output_mass, input_mass_excl, cum_stock, input_tag_incl, conv_factor)
            raise ValueError(f"Balance factor is NaN")
        elif np.abs(balance_factor - 1) > MAX_TOL_BALANCE:
            raise ValueError(
                f"Balance factor is {balance_factor:.2f}, which is out of range. Check mass balance"
            )
        # Correct included input tags
        data[input_tag_incl] = balance_factor * data[input_tag_incl]
        # Update balance factor for each belt
        for cbelt in [cb for cb in self.in_cbelts if cb not in exclude_in_cbelt]:
            cbelt.balance_factor_history.update(
                {date: balance_factor for date in data.index}
            )
            # If this cbelt as an in_belt, update cbelt upstream as well
            if cbelt.in_cbelt is not None:
                cbelt.in_cbelt.balance_factor_history.update(
                    {date: balance_factor for date in data.index}
                )
                # Update tonnage
                data[cbelt.in_cbelt.tag] = balance_factor * data[cbelt.in_cbelt.tag]
        # Return corrected data
        return data

    def _check_index_df(self, df) -> None:
        """Check that index of df is a valid time-series."""
        # Check timestamp type
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("Index in input data is not a time-series")
        # Check dates are increasing
        if not df.index.is_monotonic_increasing:
            raise ValueError(
                "Index in input data is not monotonic increasing. Sort by date first"
            )
        # Check dates are unique
        if not df.index.is_unique:
            raise ValueError("Index in input data contains repeated values")
        # Check dataset contains new dates only
        if len(df.index.intersection(set(self.stock_history.keys()))) > 0:
            raise ValueError(
                "Index in input data contains at least one date which already exists in the stock history."
                "Time overlap is not allowed between runs"
            )
