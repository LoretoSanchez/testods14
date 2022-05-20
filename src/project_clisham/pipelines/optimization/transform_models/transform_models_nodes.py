# Copyright 2020 QuantumBlack Visual Analytics Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
# OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND
# NONINFRINGEMENT. IN NO EVENT WILL THE LICENSOR OR OTHER CONTRIBUTORS
# BE LIABLE FOR ANY CLAIM, DAMAGES, OR OTHER LIABILITY, WHETHER IN AN
# ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF, OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
# The QuantumBlack Visual Analytics Limited ("QuantumBlack") name and logo
# (either separately or in combination, "QuantumBlack Trademarks") are
# trademarks of QuantumBlack. The License does not grant you any right or
# license to the QuantumBlack Trademarks. You may not use the QuantumBlack
# Trademarks or any confusingly similar mark as a trademark for your product,
# or use the QuantumBlack Trademarks in any other manner that might cause
# confusion in the marketplace, including but not limited to in advertising,
# on websites, or on software.
#
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
from typing import Any, List, Tuple, Dict
import pandas as pd
import numpy as np
import math
import project_clisham.pipelines.optimization.transform_models.reaction_updates as reaction
from project_clisham.pipelines.data_engineering.features.features_nodes import (
    create_target_flot,
    create_cuf_feature,
)

logger = logging.getLogger(__name__)


def create_models_dictionary(**dict_models) -> Dict:
    """Creates  a dictionary with the model objects

    Returns:
        [type]: [description]
    """
    td = dict_models["td"]
    params = dict_models["params"]
    dict_models.pop("td")
    dict_models.pop("params")
    new_dict = {}
    for key in dict_models:
        target_col = td.select("target", key + "_target")[0]
        new_dict[target_col] = {}
        new_dict[target_col]["model_name"] = key
        new_dict[target_col]["model_obj"] = dict_models[key]
        new_dict[target_col]["range"] = params[key]["range"]

    return new_dict


class ThroughputOptimization:
    def __init__(self, dict_models):
        self._tph = dict_models
        self.cuf = False
        self.tags_recommend = []

    def predict(self, x):
        predictions = []
        for model in self._tph.values():
            predictions.append(model["model_obj"].predict(x))
        predictions = np.array(predictions)
        # Final prediciton is sum of all the predictions
        fin_pred = np.sum(predictions, 0)

        return fin_pred


def create_throughput_optimization(dict_models):
    """Creates a ThoughputOptimization Object

    Returns:
        [type]: [description]
    """
    optim_fn = ThroughputOptimization(dict_models)

    return optim_fn


class CufOptimization:
    def __init__(self, dict_models, params, td, profit_matrix, model_tph, model_rec):
        self.models = dict_models
        self.params = params
        self.model_tph_name = model_tph
        self.model_rec_name = model_rec
        self.cuf = True
        self.tags_recommend = td.select("recommend", model_tph)
        self.profit_matrix = profit_matrix
        self.td = td
        self._current_vector = None
        self._controls = None

    @property
    def current_vector(self):
        return self._current_vector

    @current_vector.setter
    def current_vector(self, value):
        self._current_vector = value

    @property
    def controls(self):
        return self._controls

    @current_vector.setter
    def controls(self, value):
        self._controls = value
    
    def _optimization_logics(self, x):
        """
        This method is a helper function for improving the solver performance
        in order to apply some logics at the moment the mutation is taking in place.
        Examples: 
            * Recommend one or another variables, but not both.
            * Recommend multiples variables if the changes are in the same direction, 
            i.e: both increasing or both decreasing.
        """

        if self.model_tph_name == "ma1":
            x = self._bhc_and_water_logics_a1(x)
            return x
        else:
            return x

    def predict(self, x, update_values=True):
        """Combine tph and recovery models in a nested way"""
        params = self.params
        params_t = params[f"{self.model_rec_name}_flotation_target"]

        x = self._optimization_logics(x)

        _, new_x, _ = self.predict_tph(x, update_values)

        _, new_x2, _ = self.predict_rec(new_x, update_values)

        # Update CuF value
        cuf_feature = create_cuf_feature(params, new_x2, self.model_rec_name)

        # Return CuF
        return cuf_feature[params_t["cuf_obj_name"]]

    def predict_tph(self, x, update_values=True):
        params = self.params
        params_t = params[f"{self.model_rec_name}_flotation_target"]
        mod_tph_tags = params_t["tph_tags"]
        sum_tph_name = params[f"{self.model_tph_name}_target_name"]
        new_x = x.copy()
        """Combine tph and recovery models in a nested way"""
        # Update cosas molienda
        for key in mod_tph_tags:
            # TODO: warning or stop when model missing
            if key not in self.models:
                #     # TODO: warning or stop when model missing
                #     # logger.warning(f"No model has been provided for {key}")
                #     # raise RuntimeError(f"No model has been provided for {key}")
                continue
            else:
                cur_model = self.models[key]["model_obj"]
                range_v = self.models[key]["range"]
                cond = (x[key] > range_v[0]) & (x[key] < range_v[1])
                new_x.loc[cond, key] = cur_model.predict(x[cond])
        # TODO: avoid making hardcoded updates
        new_x["calc_tph_s16_over_s17"] = new_x[mod_tph_tags[0]] / new_x[mod_tph_tags[1]]
        new_x[sum_tph_name] = new_x[mod_tph_tags].sum(axis=1)
        if update_values:
            new_x = _recalculate_variables(params, new_x, self.td)
            new_x = self._update_profit_variables(new_x)
        return new_x[sum_tph_name], new_x, sum_tph_name

    def predict_rec(self, x, update_values=True):
        params = self.params
        params_t = params[f"{self.model_rec_name}_flotation_target"]
        leyes_cola = params_t["ley_cola_tf_tags"]
        recup_w = params_t["recup_wt_name"]
        new_x = x.copy()
        # Update cosas flotacion
        for key in leyes_cola:
            if key not in self.models:
                logger.warning(f"No model has been provided for {key}")
                # raise RuntimeError(f"No model has been provided for {key}")
            else:
                cur_model = self.models[key]["model_obj"]
                range_v = self.models[key]["range"]
                cond = (x[key] > range_v[0]) & (x[key] < range_v[1])
                new_x.loc[cond, key] = cur_model.predict(x[cond])

        # Update all recovery variables (by line and weighted)
        rec_features = create_target_flot(params, new_x, self.model_rec_name)
        new_x[rec_features.columns] = rec_features.values
        return new_x[recup_w], new_x, recup_w
    
    def _bhc_and_water_logics_a1(self,x):
        """
        Allows changes in bhc and water flows only if both changes
        are in the same direction. Apply this only if both are in control
        tags.
        """
        controls = self.controls
        curr = self.current_vector
        params = self.params.get("ca1xl.recommend")["bhc_and_water_logics"]
        for line in params.values():
            both_controls = (line["bhc"] in controls) and (line["water"] in controls)
            curr_bhc_val = curr[line["bhc"]].values[0]
            curr_water_val = curr[line["water"]].values[0]
            if both_controls and not pd.isna(curr_bhc_val) and not pd.isna(curr_water_val):
                # Obtaining the direction of the change
                diff_bhc = np.where(
                    x[line["bhc"]] - curr_bhc_val > 0, 1, -1
                    )
                diff_water = np.where(
                    x[line["water"]] - curr_water_val > 0, 1, -1
                    )

                # Evaluate if both directions are the same:
                # if True, leave the new values, if not, set the originals
                for tag in ["bhc", "water"]:
                    x[line[tag]] = np.where(
                        diff_bhc + diff_water!=0, x[line[tag]], curr[line[tag]].values[0]
                    )
        return x

    def fixing_ganancia_feeders_recomendations(
        self, current_value: pd.DataFrame, controls:List[str]
    ) -> pd.DataFrame:
        """
        The idea of this method is to do some fixes and validations to the
        ganancia de feeders recomendations.

        Purpose FIRST PART: "ROUNDING VALUES OF FEEDERS RECOMMENDATION"

            to round any optimized value for any ganancia de feeder tag in the
        current_value (score) dataframe.

        Purpose SECOND PART: "VALIDATING SIGNIFICATIVE CHANGES IN FEEDERS"

            the idea here is to check that the average absolute change in all
        recommended feeder in current_value (score) is greater than a fixed
        parameter (5%, i.e.).

        Args:
            -) current_value [pd.DataFrame]: is a dataframe that has two rows
            with the actual value and the recommended set of values. Its colums
            are all the tags related to the process.
            - controls [list]: the list of control tags.

        Returns:
            -) The same 'current_value' dataframe with some corrections (if
            apply).

        """

        # ######################################### #
        # ROUNDING VALUES OF FEEDERS RECOMMENDATION #
        # ######################################### #

        # List of ganancia de feeders for sag16 and sag17:
        s16_gan_feed_tags = self.params["ma2_feeders_norm"]["s16_gan_"]
        s17_gan_feed_tags = self.params["ma2_feeders_norm"]["s17_gan_"]

        # Initializing the list of tags per s16 and s17:
        s16_gan_feed_availables = list(
            set(s16_gan_feed_tags).intersection(set(controls))
        )
        s17_gan_feed_availables = list(
            set(s17_gan_feed_tags).intersection(set(controls))
        )

        # Initializing the list of Ganancia de Feeders list:
        list_gan_Feeds = [s16_gan_feed_availables, s17_gan_feed_availables]

        # Cycling over all the list of ganancia de feeders:
        for list_sag_k in list_gan_Feeds:
            # Checkig if any of these tags is in the dataset:
            if len(list_sag_k) > 0:
                # Iterating and rounding the values:
                for tag_j in list_sag_k:
                    current_value.loc["opt", tag_j] = round(
                        current_value.loc["opt", tag_j]
                    )

        # ########################################### #
        # VALIDATING SIGNIFICATIVE CHANGES IN FEEDERS #
        # ########################################### #

        # Minimal absolute average change to accept ganancia de feeders recommendations:
        threshold = self.params["ma2_feeders"]["threshold_min_sign_change"]

        # Cycling over all the list of ganancia de feeders:
        for list_sag_k in list_gan_Feeds:
            # Checkig if there is any ganancia of feeder available:
            if len(list_sag_k) > 0:
                # Initializing the average:
                average_change = 0.0
                denominator = 0

                # Geting the sum of the available ones:
                for tag_j in list_sag_k:
                    optimized = current_value.loc["opt", tag_j]
                    real = current_value.loc["curr", tag_j]

                    # Validating that the real and optimized values are numerical and
                    # non negative:
                    if (
                        (not math.isnan(real))
                        and (not math.isnan(optimized))
                        and (real > 0)
                        and (optimized > 0)
                    ):
                        denominator += 1
                        average_change += abs(optimized / real - 1.0)

                # Validating if there is something to average:
                if denominator > 0:
                    # CASE: There is at least one ganancia de feeder recommended to analyze

                    # Finishing the average change:
                    average_change = average_change / denominator

                    # Validating if they're worthy to recommend:
                    if average_change < threshold:
                        # Setting OPTIMAL equal as CURRENT so they'll be dropped in further step:
                        for tag_kl in list_sag_k:
                            current_value.loc["opt", tag_kl] = current_value.loc[
                                "curr", tag_kl
                            ]
                else:
                    # CASE: Any ganancia of feeder is available to recommend:

                    # Setting OPTIMAL equal as CURRENT so they'll be dropped in further step:
                    for tag_kl in list_sag_k:
                        current_value.loc["opt", tag_kl] = current_value.loc[
                            "curr", tag_kl
                        ]

        # Returning the output:
        return current_value

    def apply_on_off_to_tph_recommendations(self, x:pd.DataFrame)->pd.DataFrame:
        """The purpose of this method is to apply the on_off_dependencies
        to all the TPH models that are used in this optimization process.

        The original method used as inspiration of this is the _get_on_features.

        The idea is to validate if the line of the plan is working (with on_off_linea
        features) and set the optimized value equal as the current, so those
        associated recommendations will be filtered in the optimize method.

        Args:
            x: the scores dataframe being used in the optimize method.

        Returns:
            x: The same 'current_value' dataframe with some corrections (if
            apply).

        """

        # Getting the function name:
        func_name = "apply_on_off_to_tph_recommendations"

        # Getting the models for TPH:
        params = self.params
        params_t = params[f"{self.model_rec_name}_flotation_target"]
        mod_tph_tags = params_t["tph_tags"]

        # Making a copy of the input data:
        new_x = x.copy()

        # Cycling over the TPH models:
        for key in mod_tph_tags:
            # TODO: warning or stop when model missing
            if key not in self.models:
                #     # TODO: warning or stop when model missing
                #     # logger.warning(f"No model has been provided for {key}")
                #     # raise RuntimeError(f"No model has been provided for {key}")
                continue
            else:
                # Getting the inner model:
                cur_model = self.models[key]["model_obj"]

                # Getting the input tags:
                input_tags_cm = cur_model[0].selected_columns

                # Get the target of this model:
                target_cm = key

                # List of tags to "turn-off":
                list_tags_to_turn_off = input_tags_cm + [target_cm]

                # Cycling over all the tags
                for feature in list_tags_to_turn_off:
                    on_flag = all(
                        [
                            new_x.loc["curr", d] > 0.5
                            for d in self.td.dependencies(feature)
                        ]
                    )
                    if on_flag:
                        if np.isnan(x.loc["curr", feature]):
                            # Printing that the actual feature is NAN:
                            logger.warning(
                                f"Current Value for feature {feature} is NaN " +
                                f"@ {func_name}."
                            )

                            # Setting "opt" as "curr" so it will be discarded:
                            new_x.loc["opt", feature] = new_x.loc["curr", feature]
                        # else:
                        #    on_controls.append(feature)
                    else:
                        # Printing that the actual feature is OFF:
                        logger.warning(
                            f"Current feature {feature} is OFF "+
                            f"@ {func_name}."
                        )

                        # Setting "opt" as "curr" so it will be discarded:
                        new_x.loc["opt", feature] = new_x.loc["curr", feature]

        # Returning the modified scores dataframe:
        return new_x

    def correct_pHs_recommended_plant(self, x:pd.DataFrame, controls:List[str])->pd.DataFrame:
        """The purpose is to validate the pH recommendations in the flotation plants.
        Two checks are applied: first, if both pH variables are in the controls list; and second, if the flotation lines are active.
        - If both flotation lines are active, then the pH is replaced with the average measurement of both sensors; if not,
        the value of that unique sensor is maintained as the off line is not representative of the operation reality.
        - If only one of the pH sensors is in controls, then that value is replaced with the average of both, and the other optimized value is replaced
        by the current value (opt = curr), thus discarding the recommendation.

        Args:
            x: the scores dataframe being used in the optimize method.
            controls: list of controllable features
        Returns:
            x: The same 'current_value' dataframe with some corrections (if
            apply).

        """
        # Getting the 'validate_calibrated_phs_sensors':
        clbrtn_params = {
            'fa0': self.params["ca0xl.recommend"]["validate_calibrated_phs_sensors"],
            'fa1': self.params["ca1xl.recommend"]["validate_calibrated_phs_sensors"],
            'fa2': self.params["ca2xl.recommend"]["validate_calibrated_phs_sensors"]}

        for plant in clbrtn_params:
        
            #check if there is a list of pH sensors in the corresponding plant: if not, ignore the plant and continue to the next one.
            if not clbrtn_params[plant]["tags"]:
                continue
            

            #Get the on_off criteria for each flotation line
            average_rotors_on_1=clbrtn_params[plant]["rotor_tags"][0]
            average_rotors_on_2=clbrtn_params[plant]["rotor_tags"][1]

            #If there are less rotors working than the threshold, then the line is considered off;
            #thus, the pH value of that line would not be representative of the entire process, and should not be considered in the average
            sensor_1_on=True
            sensor_2_on=True
            if x.loc["curr", average_rotors_on_1] <= clbrtn_params[plant]["rotor_threshold"]:
                sensor_1_on=False
                logger.warning(f"{plant} line 1 is out of service according to rotor limitation. excluding its consideration on the pH recommendation")
            
            if x.loc["curr", average_rotors_on_2] <= clbrtn_params[plant]["rotor_threshold"]:
                sensor_2_on=False
                logger.warning(f"{plant} line 2 is out of service according to rotor limitation. excluding its consideration on the pH recommendation")


            # Get the pH tag names:
            sensor_1 = clbrtn_params[plant]["tags"][0]
            sensor_2 = clbrtn_params[plant]["tags"][1]

            # Getting the optimized and current values in 'x':
            sensor_1_opt = x.loc["opt", sensor_1]
            sensor_2_opt = x.loc["opt", sensor_2]
            sensor_1_curr = x.loc["curr", sensor_1]
            sensor_2_curr = x.loc["curr", sensor_2]


            # VALIDATION:
            if (sensor_1 in controls) and (sensor_2 in controls):
                # ################################### #
                # CASE: Both sensors were recommended #
                # ################################### #

                #Then we should estimate the average value, only if both lines are on
                if sensor_1_on and sensor_2_on:
                # Getting the averages:
                    average_ph_opt = (sensor_1_opt + sensor_2_opt) / 2.0
                    average_ph_curr = (sensor_1_curr + sensor_2_curr) / 2.0

                    # Updating the first pH current:
                    x.loc["curr", sensor_1] = average_ph_curr

                    # Updating the first pH optimized:
                    x.loc["opt", sensor_1] = average_ph_opt

                    # Assigning the cur == opt so there's no recommendation here:
                    x.loc["opt", sensor_2] = x.loc["curr", sensor_2]

                elif sensor_1_on:
                    # if we only have one line active: we should maintain the current value and
                    # stop recommending the other line

                    # Assigning the cur == opt so there's no recommendation in the second sensor:
                    x.loc["opt", sensor_2] = x.loc["curr", sensor_2]

                elif sensor_2_on:
                    # then we only have one line active: we should maintain the current value and
                    # stop recommending the other line

                    # Assigning the cur == opt so there's no recommendation in the first sensor:
                    x.loc["opt", sensor_1] = x.loc["curr", sensor_1]
                else:
                    #if both lines are off, we shouldn't recommend any of them
                    x.loc["opt", sensor_1] = x.loc["curr", sensor_1]
                    x.loc["opt", sensor_2] = x.loc["curr", sensor_2]



            elif (sensor_1 in controls) and (sensor_2 not in controls):
                # ############################################### #
                # CASE: Only recommendation with the first sensor #
                # ############################################### #

                #for sensor 1, we should still use the second sensor and get the average, if both lines are on
                if sensor_2_on and sensor_1_on:
                    average_ph_opt = (sensor_1_opt + sensor_2_opt) / 2.0
                    average_ph_curr = (sensor_1_curr + sensor_2_curr) / 2.0

                if not(sensor_1_on):
                    #if the line is off, then we shouldn't recommend the line
                    x.loc["opt", sensor_1] = x.loc["curr", sensor_1]

                #any other case, we should use the sensor 1's original value.


                # Assigning the cur == opt so there's no recommendation in the second sensor:
                x.loc["opt", sensor_2] = x.loc["curr", sensor_2]

            elif (sensor_1 not in controls) and (sensor_2 in controls):
                # ################################################ #
                # CASE: Only recommendation with the second sensor #
                # ################################################ #


                #for sensor 2, we should still use the second sensor and get the average, if both lines are on
                if sensor_2_on and sensor_1_on:
                    average_ph_opt = (sensor_1_opt + sensor_2_opt) / 2.0
                    average_ph_curr = (sensor_1_curr + sensor_2_curr) / 2.0

                if not(sensor_2_on):
                    #if the line is off, then we shouldn't recommend the line
                    x.loc["opt", sensor_2] = x.loc["curr", sensor_2]

                #any other case, we should use the sensor 2's original value.


                # Assigning the cur == opt so there's no recommendation in the first sensor:
                x.loc["opt", sensor_1] = x.loc["curr", sensor_1]
            else:
                # ######################################### #
                # CASE: none of the sensors are in controls #
                # ######################################### #

                # Assigning the cur == opt so there's no recommendation
                # in both sensors:
                x.loc["opt", sensor_1] = x.loc["curr", sensor_1]
                x.loc["opt", sensor_2] = x.loc["curr", sensor_2]

        # Returning the corrected dataframe:
        return x


    def recommend_profit_mv_out_of_bounds(self, x:pd.DataFrame, controls: List[str]) -> pd.DataFrame:
        """The main idea of this method is to validate if the recommended values
        of the profit mv 'molienda' tags are outside actual profit bounds. If the're
        inside, then do not recommend, if not, go ahead and recommend.

        Args:
            x: the scores dataframe being used in the optimize method.
            controls: list of controllable features

        Returns:
            x: The same 'current_value' dataframe with some corrections (if
            apply).

        """

        # Getting the function name:
        funct_name = "recommend_profit_mv_out_of_bounds"

        # Getting the 'not_recommend_mv_in_range' feature of all plants:
        dict_ranges_profit_mv = {
            "ma2": self.params["ma2.recommend"]["not_recommend_mv_in_range"],
            "ca0xl": self.params["ca0xl.recommend"]["not_recommend_mv_in_range"],
            "ca1xl": self.params["ca1xl.recommend"]["not_recommend_mv_in_range"],
            "ca2xl": self.params["ca2xl.recommend"]["not_recommend_mv_in_range"],
        }

        # Initializing a list that will contain if any control profit mv
        # tag will be not recommended:
        list_control_tags_dropped = []

        # Cycling over all the plant parameter:
        for plant_k in dict_ranges_profit_mv:
            # Cycling over all the profit mv 'molienda' tag:
            for profit_vm_molienda in dict_ranges_profit_mv[plant_k]:
                # Validating if this tags is in controls, so the check can be done:
                if profit_vm_molienda in controls:
                    # Getting the limits:
                    profit_low_lmt = dict_ranges_profit_mv[plant_k][profit_vm_molienda][
                        0
                    ]
                    profit_hig_lmt = dict_ranges_profit_mv[plant_k][profit_vm_molienda][
                        1
                    ]

                    # Getting the values of the tag and limits SUGGESTED:
                    tag_val_opt = x.loc["opt", profit_vm_molienda]
                    tag_val_lolmt_opt = x.loc["opt", profit_low_lmt]
                    tag_val_hilmt_opt = x.loc["opt", profit_hig_lmt]

                    # Validating if the suggested value lies outside the
                    # actual limits so it could be turned off:
                    if (tag_val_lolmt_opt <= tag_val_opt) and (
                        tag_val_opt <= tag_val_hilmt_opt
                    ):
                        # CASE: Recommended value is outside limites -> NotRecommend
                        x.loc["opt", profit_vm_molienda] = x.loc[
                            "curr", profit_vm_molienda
                        ]

                        # Saving the tag turned off:
                        list_control_tags_dropped.append(profit_vm_molienda)

        # Printing the resume if some tags has been turned off:
        if len(list_control_tags_dropped) > 0:
            logger.warning(
                f"The tags {list_control_tags_dropped} has been " +
                f"removed from recommendations because the suggested " +
                f"values lies inside the Profit Limits @{funct_name}."
            )

        # Returning the modified x dataframe:
        return x

    #TODO: the filter should limit the optimization range, thus being applied before the optimization, not after.
    def recommend_ph_chance_open_lime_valves(self, x:pd.DataFrame, controls: List[str]) -> pd.DataFrame:
        """This function will limit the recommendation of the pH tags if the lime valves are open over the maximum thresholds. limits the maximum recommended value for the pH valves, based on a maximum threshold.
        It checks two conditions: that the optimized valve value is higher than the current; and if the recommended
        value is higher than the threshold.

        Args:
            x: the scores dataframe being used in the optimize method.
            controls: : list of controllable features

        Returns:
            x: The same 'current_value' dataframe with some corrections (if apply).
        """

        # Getting the function name:
        funct_name = "recommend_ph_chance_open_lime_valves"

        # Getting the 'not_recommend_lime_valvs_full' feature of all plants:
        dict_lime_valves_params = {
            "ca0xl": self.params["ca0xl.recommend"]["not_recommend_lime_valvs_full"],
            "ca1xl": self.params["ca1xl.recommend"]["not_recommend_lime_valvs_full"],
            "ca2xl": self.params["ca2xl.recommend"]["not_recommend_lime_valvs_full"],
        }

        # Initializing a boolean variable to check if pH is not recommended:
        ph_tag_not_recommended = False

        # Initializing the list of tags not recommended:
        list_tag_not_recommended = []

        # Check for each of the plants:
        for plant_k in dict_lime_valves_params:
            # Getting the params for this plant:
            params_plant_k = dict_lime_valves_params[plant_k]

            # Getting the number of lime valves set:
            num_lvs = len(params_plant_k["lime_valvs_tags"])

            # Validating there are valid parameters set for this plant:
            if (num_lvs > 0) and (params_plant_k["ph_line_tag"] is not None):
                # CASE:
                #       Parameters set for this plant.

                # Getting the pH tag:
                ph_tag = params_plant_k["ph_line_tag"]

                # Checking if the pH tag is a control tag for this shift optimization:
                if ph_tag in controls:
                    # Getting the respective threshold:
                    th_param = params_plant_k["lime_valvs_th"]

                    # Getting the current and optimized value for this tag:
                    real_ph_Tagvalue = x.loc["curr", ph_tag]
                    opt_ph_Tagvalue = x.loc["opt", ph_tag]

                    # Then validating an increment in this potential recommendation:
                    cond1 = opt_ph_Tagvalue > real_ph_Tagvalue

                    # Validating if the lime valves values are higher than the
                    # set threshold:
                    cond2 = True
                    for lm_tagj in params_plant_k["lime_valvs_tags"]:
                        # Getting the numerical value:
                        lm_tagj_value = x.loc["curr", lm_tagj]

                        # Checking if the value is higher than the threshold:
                        if lm_tagj_value < th_param:
                            cond2 = False

                    # Validating all conditions:
                    if cond1 & cond2:
                        # NOT RECOMEND pH:
                        x.loc["opt", ph_tag] = x.loc["curr", ph_tag]

                        # Updating boolean variable:
                        ph_tag_not_recommended = True

                        # Augmenting the list of tags not recommended:
                        list_tag_not_recommended.append(ph_tag)
        
        # Printing the resume if some tags has been turned off:
        if ph_tag_not_recommended:
            plurals = ""
            if len(list_tag_not_recommended) > 1:
                plurals = "s"
            logger.warning(
                            f"The tag{plurals} {list_tag_not_recommended} " +
                            f"has been removed from recommendations because " +
                            f"associated lime valves are close to its upper " +
                            f"limit @{funct_name}."
                        )
        # Returning the modified x dataframe:
        return x


    def constraint_malla_range(self, x:pd.DataFrame)->pd.DataFrame:
        """The purpose of this method is to restrict the range of "Malla" recommendations requested by the JdU.

        According to the JdU, "Malla" should be recommended only if the value is greater than 24. 
        If it is 24<malla<33, it should be recommended to increase the value. 
        If it is malla>=33, it should be recommended to decrease the value.

        Args:
            x: the scores dataframe being used in the optimize method.

        Returns:
            x: The restricted 'current_value' dataframe.
        """

        params_malla = self.params["ca1xl.recommend"]["malla_range"]

        # Getting the function name:
        funct_name = "constraint_malla_range"

        # Getting the tag names:
        malla_seccion = params_malla["tags_malla"]
        min_range = params_malla['min_range']
        max_range = params_malla['max_range']
        
        # Getting the optimized and current values in 'x':

        for seccion in malla_seccion:
            malla_seccion_opt = x.loc["opt", seccion]
            malla_seccion_curr = x.loc["curr", seccion]

            #Conditions  
            if malla_seccion_curr < min_range:
                # Assigning the cur == opt so there's no recommendation here:
                x.loc["opt", seccion] = x.loc["curr", seccion]

            elif (malla_seccion_curr < max_range) and (malla_seccion_opt <= malla_seccion_curr):
                # Assigning the cur == opt so there's no recommendation here:
                x.loc["opt", seccion] = x.loc["curr", seccion]

            elif (malla_seccion_curr >= max_range) and (malla_seccion_opt >= malla_seccion_curr):
                # Assigning the cur == opt so there's no recommendation here:
                x.loc["opt", seccion] = x.loc["curr", seccion]

        return x

    def constraint_malla_range_over_tph(self, x:pd.DataFrame, controls:List[str])->pd.DataFrame:
        """
        The idea of this method is not showing the TPH recommendations
        when the Malla is over their max ranges. This shouldn't affect
        the optimization process due TPH is the target of the optimization
        so if all the other variables are followed the system should be able
        to reach the TPH value in the recommendation.
        Args:
            x: the scores dataframe being used in the optimize method.
            controls: Controls tags.
        Returns:
            The restricted 'current_value' dataframe.
        """

        mallas_dict = {
            'ma0': self.params["ca0xl.recommend"]["malla_range"],
            'ma1': self.params["ca1xl.recommend"]["malla_range"]
        }

        # Getting the function name:
        model_name = self.model_tph_name
        _controls = controls.copy()
        #for planta in mallas_dict:
        if model_name in mallas_dict.keys():
            max_range = mallas_dict[model_name]['max_range']
            for i,malla in enumerate(mallas_dict[model_name]['tags_malla']):
                tph = mallas_dict[model_name]['tags_tph'][i]
                if tph in _controls and x.loc["curr", malla] > max_range:
                    _controls.remove(tph)
                    logger.warning(
                            f"Malla {malla} is over {max_range}. We are " +
                            f"skipping Tph {tph} recommendation for this shift."
                        )
        return _controls
    
    def not_recommend_tph_SAG_in_range(self, x:pd.DataFrame)->pd.DataFrame:
        """The purpose of this method is to stop recommending the TPH (SAG) when the high limit is open. 

        Args:
            x: the scores dataframe being used in the optimize method.

        Returns:
            x: The restricted 'current_value' dataframe.
        """

        params_tph_sag = self.params["ca2xl.recommend"]["not_recommend_TPH_in_range"]

        # Getting the function name:
        funct_name = "not_recommend_tph_SAG_in_range"

        # Getting the tag names:
        tags_tph = params_tph_sag["tags_tph_sag"]
        tags_HL_tph = params_tph_sag["tags_HL_tph_sag"]
        range_tph_sl = params_tph_sag['range_tph_SL']
        range_tph_min = params_tph_sag['range_tph_min']
        
        # Getting the optimized and current values in 'x':

        for sag,hl in zip(tags_tph,tags_HL_tph):
            tph_SAG_opt = x.loc["opt", sag]
            tph_SAG_curr = x.loc["curr", sag]
            tph_SAG_HL_curr = x.loc["curr", hl]

        #Conditions  
            if (tph_SAG_curr < (tph_SAG_HL_curr-range_tph_sl)) and (tph_SAG_opt <= (tph_SAG_HL_curr+range_tph_min)):
                # Assigning the cur == opt so there's no recommendation here:
                x.loc["opt", sag] = x.loc["curr", sag]
                logger.warning(f"Current tph ({sag}) recommendation has not been applied due to the PROFIT system high limit not being restricted")

        return x
    
    def not_show_tph_recs_if_tolva_conc_low(self, x:pd.DataFrame, controls:List[str])->pd.DataFrame:
        """
        The idea of this method is not show the TPH recommendations
        when the Tolva Concentradora is lower than a threshold value. 
        This shouldn't affect the optimization process due TPH is the target of the 
        optimization so if all the other variables are followed the system should 
        be able to reach the TPH value in the recommendation.
        Args:
            x: the scores dataframe being used in the optimize method.
            controls: Controls tags.
        Returns:
            Updated controls list.
        """

        tph_vs_tolva = {
            'ma0': self.params["ca0xl.recommend"]["tph_vs_tolva"],
            'ma1': self.params["ca1xl.recommend"]["tph_vs_tolva"]
        }

        # Getting the function name:
        model_name = self.model_tph_name
        _controls = controls.copy()
        #for planta in mallas_dict:
        if model_name in tph_vs_tolva.keys():
            min_range = tph_vs_tolva[model_name]['min_range']
            tolva_con_tag = tph_vs_tolva[model_name]['tags_tolva']
            actual_tolva_con = x.loc["curr", tolva_con_tag]
            if  actual_tolva_con < min_range:
                _controls = [x for x in _controls if x not in tph_vs_tolva[model_name]['tags_tph']]
                tph_skipped = [x for x in controls if x not in _controls]
                if tph_skipped:
                    logger.warning(
                        f"TPHs {','.join(tph_skipped)} were removed from the recommendation " +
                        f"set due {tolva_con_tag} is lower than {min_range} (actual: {actual_tolva_con})." 
                    )
        return _controls
    
    def _update_profit_variables(self, x_vector):
        if self.current_vector is None:
            raise ValueError(
                "Current scores should be initialized before running optimization"
            )
        else:
            cur_row = self.current_vector.copy()

            diff_df = x_vector.copy()
            new_x_vector = x_vector.copy()

            com_cols = set(diff_df.columns).intersection(cur_row.columns)
            for col in ["run_id", "fecha"]:
                if col in com_cols:
                    com_cols.remove(col)
            diff_df = diff_df[com_cols]
            diff_df = diff_df.subtract(cur_row[com_cols].values, axis="columns")
            profit_matrix = self.profit_matrix

            for var in profit_matrix["tags"]:

                if var in diff_df.columns:
                    change = pd.DataFrame([0] * len(diff_df.index), columns=["check"])
                    for col in profit_matrix.columns[1:]:
                        if col in diff_df.columns:
                            value = profit_matrix.loc[
                                profit_matrix["tags"] == var, col
                            ].values[0]
                            change["check"] = (
                                change["check"].values
                                + (diff_df[col].fillna(0) * value).values
                            )
                            change["check"].fillna(0, inplace=True)
                        else:
                            raise ValueError(
                                f"The variable {col} from profit matrix is not in the input DF"
                            )
                    new_x_vector[var] = (
                        new_x_vector[var].values + change["check"].values
                    )
                else:
                    raise ValueError(
                        f"The variable {var} from profit matrix is not in the input DF"
                    )
            x_vector[new_x_vector.columns] = new_x_vector
        return x_vector


def create_cuf_optimization_a2(dict_models, params, td, profit_matrix):
    """Creates a CufOptimization Object

    Returns:
        [type]: [description]
    """
    optim_fn = CufOptimization(dict_models, params, td, profit_matrix, "ma2", "fa2")

    return optim_fn


def create_cuf_optimization_a1(
    dict_models, params, td, profit_matrix
):  # TODO: make modular function
    """Creates a CufOptimization Object

    Returns:
        [type]: [description]
    """
    optim_fn = CufOptimization(dict_models, params, td, profit_matrix, "ma1", "fa1")

    return optim_fn


def create_cuf_optimization_a0(
    dict_models, params, td, profit_matrix
):  # TODO: make modular function
    """Creates a CufOptimization Object

    Returns:
        [type]: [description]
    """
    optim_fn = CufOptimization(dict_models, params, td, profit_matrix, "ma0", "fa0")

    return optim_fn


def _recalculate_variables(params, new_x, td):
    """
    Wrapper around all reaction function used in DCH otpimization
    process.
    """
    params_recalculate = params["recalculate"]
    for fun in params_recalculate:
        params_vars = params_recalculate[fun]
        resp = getattr(reaction, fun)(params, new_x, td)
        for var in params_vars:
            new_x[var] = resp[var]
    return new_x
