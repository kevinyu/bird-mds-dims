import glob
import os
import re

import numpy as np
import h5py


class DataExplorer(object):
    def __init__(self,
            data_dir,
            bird_name,
            site_num,
            exclude_song=False,
            exclude_noise=False):
        self.bird_dir = os.path.join(data_dir, bird_name)
        self._site_prefix = "Site{}".format(site_num)
        self.exclude_song = exclude_song
        self.exclude_noise = exclude_noise

        self._single_unit_re = re.compile(os.path.join(self.bird_dir, 
            "{}_.+_e"
            "(?P<electrode_id>\d+)"
            "_s\d+_ss"
            "(?P<unit_id>\d+).h5".format(self._site_prefix)))
        self._multi_unit_re = re.compile(os.path.join(self.bird_dir, 
            "{}_.+_e"
            "(?P<electrode_id>\d+)"
            "_s\d+.h5".format(self._site_prefix)))

        self._call = None
        for key in list(self.single_units.values())[0].keys():
            if key.startswith("Call"):
                self._call = key
                break

    def _match_unit_files(self, regex):
        unit_data = {}
        for filename in glob.glob(os.path.join(self.bird_dir, "{}*.h5".format(self._site_prefix))):
            match = regex.match(filename)
            if not match:
                continue
            electrode_unit_tuple = tuple(map(int, match.groups()))
            unit_data[electrode_unit_tuple] = h5py.File(
                    os.path.join(self.bird_dir, match.group()), "r", libver="latest")
        return unit_data

    @property
    def single_units(self):
        """Return paths to all h5 files of spike sorted units for this site"""
        return self._match_unit_files(self._single_unit_re)

    def _stim_to_stim_type(self):
        stim_ids = self._stim_ids()
        stim_types = self.stim_types()

        result = {}
        for stim in self.stims():
            callid = stim_ids[stim].attrs.get("callid")
            # > Since callids are sorted alphabetically, DC and Di are
            #   usually next to each other but tend to have similar responses
            #   the below is a hack to make DC and Di
            #   further apart so they are colored differently in visualizations
            # 
            # if callid == "Di":
            #     callid = "di"
            stim_type_key = (
                    stim_ids[stim].attrs.get("stim_type"),
                    callid)
            stim_type_idx = stim_types.index(stim_type_key)
            result[stim] = stim_type_idx
        return result

    def _stim_ids(self):
        result = {}
        stim_ids = self.single_units.values()[0][self._call]
        for stim_id in stim_ids:
            if self.exclude_song and stim_ids[stim_id].attrs.get("stim_type") == "song":
                continue
            elif self.exclude_noise and stim_ids[stim_id].attrs.get("stim_type") == "mlnoise":
                continue
            else:
                result[stim_id] = stim_ids[stim_id]
        return result

    def stims(self):
        return sorted(self._stim_ids().keys())

    def stim_types(self):
        stim_types = set()
        stim_ids = self._stim_ids()
        for stim in stim_ids:
            callid = stim_ids[stim].attrs.get("callid")
            # if callid == "Di":
            #     callid = "di"
            stim_types.add((
                stim_ids[stim].attrs.get("stim_type"),
                callid))
        return sorted(stim_types)

    def load_table(self, filter_unit=None, load_spike_times=False):
        stim_to_stim_type = self._stim_to_stim_type()
        stims = self.stims()

        table = []
        spike_time_arr = []
        for stim_idx, stim in enumerate(stims):
            stim_type_idx = stim_to_stim_type[stim]
            for (electrode_id, unit_id), unit in self.single_units.items():
                if filter_unit and (electrode_id, unit_id) != filter_unit:
                    continue
                for trial_num in unit[self._call][stims[stim_idx]]:
                    trial_idx = int(trial_num) - 1
                    table.append((
                        electrode_id,
                        unit_id,
                        stim_idx,
                        stim_type_idx,
                        trial_idx))
                    if load_spike_times:
                        spike_data = unit[self._call][stim][trial_num]
                        if spike_data["spike_times"][0] == -999:
                            spike_times = np.array([[]])
                        else:
                            spike_times = spike_data["spike_times"][:]
                        spike_time_arr.append(spike_times)

        table = np.array(table, dtype=[
            ("electrode", "|i4"),
            ("unit", "|i4"),
            ("stim", "|i4"),
            ("stim_type", "|i4"),
            ("trial", "|i4")])

        if load_spike_times:
            return table, spike_time_arr
        else:
            return table

    def stim_sorter(self, table):
        return np.argsort(zip(*sorted(set(zip(table["stim"], table["stim_type"]))))[1])

