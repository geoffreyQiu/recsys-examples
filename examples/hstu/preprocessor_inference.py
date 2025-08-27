# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import logging
import os
import sys
import tarfile
from typing import Dict, List
from urllib.request import urlretrieve
from zipfile import ZipFile

import bisect

import numpy as np
import pandas as pd

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
log = logging.getLogger("main")

import time
import datetime


def reporthook(count, block_size, total_size):
    global start_time
    if count == 0:
        start_time = time.time()
        return
    duration = time.time() - start_time
    progress_size = int(count * block_size)
    speed = int(progress_size / (1024 * duration))
    percent = int(count * block_size * 100 / total_size)
    sys.stdout.write(
        "\r...%d%%, %d MB, %d KB/s, %d seconds passed"
        % (percent, progress_size / (1024 * 1024), speed, duration)
    )
    sys.stdout.flush()


def _one_hot_encode(row):
    mapping = {category: i + 1 for i, category in enumerate(row.unique())}
    row = row.map(mapping)
    return row


class DataProcessor:
    """
    Abstract class for data processing.

    Args:
        download_url (str): URL from which to download the dataset.
        data_path (str): Path where the dataset will be stored.
        file_name (str): Name of the file containing the dataset.
        prefix (str): The root directory of the dataset.
    """

    def __init__(
        self,
        download_url: str,
        data_path: str,
        file_name: str,
        prefix: str,
    ) -> None:
        self._download_url = download_url
        self._data_path = data_path
        os.makedirs(self._data_path, exist_ok=True)
        self._file_name = file_name
        self._prefix = prefix
    
    def _post_process_use_splits(
        self,
        user_feature_df,
        sequence_feature_df,
        batching_df,
        user_id_feature_name: str,
        contextual_feature_names: List[str],
        item_feature_name: str,
        action_feature_name: str,
        sequence_file,
        batching_file,
    ) -> None:
        if user_feature_df is not None:
            final_df = pd.merge(
                sequence_feature_df, user_feature_df, on=user_id_feature_name
            )
        else:
            final_df = sequence_feature_df
        final_df.to_csv(sequence_file, index=False, sep=",")
        batching_df.to_csv(batching_file, index=False, sep=",")
        log.info(f"Processed file saved to {sequence_file} and {batching_file}")
        log.info(f"num users: {len(final_df[user_id_feature_name])}")
        data = []
        for name in contextual_feature_names:
            data.append([name, final_df[name].min(), final_df[name].max()])
        log.info(["feature_name", "min", "max"])
        log.info(data)
        
    def _post_process(
        self,
        user_feature_df,
        sequence_feature_df,
        user_id_feature_name: str,
        contextual_feature_names: List[str],
        item_feature_name: str,
        action_feature_name: str,
        output_file,
    ) -> None:
        if user_feature_df is not None:
            final_df = pd.merge(
                sequence_feature_df, user_feature_df, on=user_id_feature_name
            )
        else:
            final_df = sequence_feature_df
        final_df = final_df[:2500]
        
        
        time_interval = 60000
        def group_timestamp_by_interval(ts):
            idx = 0
            ts_indptr = list()
            while idx < len(ts):
                idx = bisect.bisect_right(ts, ts[idx]+time_interval, lo=idx)
                ts_indptr.append(idx)
            return ts_indptr
        
        def group_history_by_timestamp(seq, ts_indptr):
            return [ list(seq[:ts_end]) for ts_end in ts_indptr ]
        
        # break on timestamps:
        split_df = None
        num_samples = len(final_df)
        for idx in range(num_samples):
            if idx % 50 == 0:
                print("Finished", 100. * idx/num_samples , "% samples" )
            data = final_df.iloc[idx]
            ts = data["time_ms"]
            video_id = data["video_id"]
            action_weights = data["action_weights"]
            play_time_ms = data["play_time_ms"]
            duration_ms = data["duration_ms"]
            user_history = zip(ts, video_id, action_weights, play_time_ms, duration_ms)
            user_history = sorted(user_history, key=lambda item: item[0])
            ts, video_id, action_weights, play_time_ms, duration_ms = zip(*user_history)

            ts_indptr = group_timestamp_by_interval(ts)
            batch_ts = group_history_by_timestamp(ts, ts_indptr)
            batch_video_id = group_history_by_timestamp(video_id, ts_indptr)
            batch_action_weights = group_history_by_timestamp(action_weights, ts_indptr)
            batch_play_time_ms = group_history_by_timestamp(play_time_ms, ts_indptr)
            batch_duration_ms = group_history_by_timestamp(duration_ms, ts_indptr)
            emit_timestamp = [ ts[idx-1] for idx in ts_indptr ]
            batch_data = {
                'user_id': [ data["user_id"] ] * len(ts_indptr),
                'time_ms': batch_ts,
                'video_id': batch_video_id,
                'action_weights': batch_action_weights,
                'play_time_ms': batch_play_time_ms,
                'duration_ms': batch_duration_ms,
                'emit_timestamp': emit_timestamp
            }
            if split_df is None:
                split_df = pd.DataFrame(data=batch_data)
            else:
                split_df = pd.concat([split_df, pd.DataFrame(data=batch_data)], ignore_index=True)
        log.info("Batched all history")

        final_df.drop(columns=['time_ms', 'video_id', 'action_weights', 'play_time_ms', 'duration_ms'], inplace=True)
        final_df = final_df.merge(split_df, on='user_id')
        final_df = final_df.sort_values(by=['emit_timestamp'], ascending=True)
        log.info("Joined all history")
        
        final_df.to_csv(output_file, index=False, sep=",")
        log.info(f"Processed file saved to {output_file}")
        log.info(f"num users: {len(final_df[user_id_feature_name])}")
        data = []
        for name in contextual_feature_names:
            data.append([name, final_df[name].min(), final_df[name].max()])
        log.info(["feature_name", "min", "max"])
        log.info(data)

        data = []
        for name in [item_feature_name, action_feature_name]:
            max_seq_len = int(final_df[name].apply(len).max())
            min_seq_len = int(final_df[name].apply(len).min())
            average_seq_len = int(final_df[name].apply(len).mean())
            min_id = int(final_df[name].apply(min).min())
            max_id = int(final_df[name].apply(max).max())
            data.append(
                [name, min_id, max_id, min_seq_len, max_seq_len, average_seq_len]
            )
        log.info(
            [
                "feature_name",
                "min",
                "max",
                "min_seqlen",
                "max_seqlen",
                "average_seqlen",
            ]
        )
        log.info(data)

    def file_exists(self, name: str) -> bool:
        return os.path.isfile("%s/%s" % (os.getcwd(), name))

class DLRMKuaiRandProcessor(DataProcessor):
    """

    Data processor for the `KuaiRand <https://kuairand.com/>`_ dataset.

    Args:
        download_url (str): URL from which to download the dataset.
        data_path (str): Path where the dataset will be stored.
        file_name (str): Name of the file containing the dataset.
        prefix (str): The root directory of the dataset.
    """

    def __init__(
        self,
        download_url: str,
        data_path: str,
        file_name: str,
        prefix: str,
    ) -> None:
        super().__init__(download_url, data_path, file_name, prefix)
        self._item_feature_name = "video_id"
        self._action_feature_name = "action_weights"
        self._contextual_feature_names = [
            "user_id",
            "user_active_degree",
            "follow_user_num_range",
            "fans_user_num_range",
            "friend_user_num_range",
            "register_days_range",
        ]
        base_path = os.path.join(data_path, prefix, "data")
        if prefix == "KuaiRand-Pure":
            self._log_files = [
                os.path.join(base_path, "log_standard_4_08_to_4_21_pure.csv"),
                os.path.join(base_path, "log_standard_4_22_to_5_08_pure.csv"),
            ]
            self._user_features_file = (
                os.path.join(base_path, "user_features_pure.csv")
            )

        elif prefix == "KuaiRand-1K":
            self._log_files = [
                os.path.join(base_path, "log_standard_4_08_to_4_21_1k.csv"),
                os.path.join(base_path, "log_standard_4_22_to_5_08_1k.csv"),
            ]
            self._user_features_file = os.path.join(base_path, "user_features_1k.csv")
        elif prefix == "KuaiRand-27K":
            self._log_files = [
                os.path.join(base_path, "log_standard_4_08_to_4_21_27k_part1.csv"),
                os.path.join(base_path, "log_standard_4_08_to_4_21_27k_part2.csv"),
                os.path.join(base_path, "log_standard_4_22_to_5_08_27k_part1.csv"),
                os.path.join(base_path, "log_standard_4_22_to_5_08_27k_part2.csv"),
            ]
            self._user_features_file = os.path.join(base_path, "user_features_27k.csv")
        self._output_file: str = os.path.join(base_path, "processed_seqs.csv")
        self._batch_file: str = os.path.join(base_path, "processed_batches.csv")
        self._event_merge_weight: Dict[str, int] = {
            "is_click": 1,
            "is_like": 2,
            "is_follow": 4,
            "is_comment": 8,
            "is_forward": 16,
            "is_hate": 32,
            "long_view": 64,
            "is_profile_enter": 128,
        }

    def download(self) -> None:
        """
        Download and decompress the dataset. The downloaded dataset will be saved in the "tmp" directory.
        """
        file_path = f"{self._data_path}{self._file_name}"
        if not self.file_exists(file_path):
            log.info(f"Downloading {self._download_url}")
            urlretrieve(self._download_url, file_path, reporthook)
            log.info(f"Downloaded to {file_path}")
            with tarfile.open(file_path, "r:*") as tar_ref:
                tar_ref.extractall(path=self._data_path)
                log.info("Data files extracted")

    def preprocess(self) -> None:
        """
        Preprocess the raw data. The support dataset are "KuaiRand-Pure", "KuaiRand-1K", "KuaiRand-27K".
        """

        self.download()
        log.info("Preprocessing data...")
        seq_cols = [
            "video_id",
            "time_ms",
            "action_weights",
            "play_time_ms",
            "duration_ms",
        ]
        time_interval = 60000
        def create_interval(group):
            sorted_group = sorted(zip(*(group[_col_name] for _col_name in seq_cols)),  key=lambda item: item[1])
            interval_group = [0]
            for idx in range(1, len(group['time_ms'])):
                delta_time = sorted_group[idx][1] - sorted_group[interval_group[-1]][1]
                if delta_time >= time_interval:
                    interval_group.append(idx)
            interval_group.append(len(group['time_ms']))
            return pd.Series(
                [ group.user_id, group.date ] + [ list(l) for l in zip(*sorted_group) ] + [ interval_group, len(interval_group)-1 ],
                index=[ 'user_id', 'date' ] + seq_cols + [ "interval_indptr", "num_intervals" ]
            )
        
        def filter_interval(group):
            idx = group.interval_indptr[int(group.interval_counter)]
            return pd.Series(
                [ group.user_id, group.date ] + [ list(group[_col_name][:idx]) for _col_name in seq_cols ] + [ group["time_ms"][idx-1], idx ],
                index=[ 'user_id', 'date' ] + seq_cols + [ "interval_end_ts", "interval_indptr" ]
            )
        
        def filter_interval2(group):
            timestamp_col_name = "time_ms"
            idx = group.interval_indptr[int(group.interval_counter)]
            return pd.Series(
                [ group.user_id, group.date, group["time_ms"][idx-1], idx ],
                index=[ 'user_id', 'date', "interval_end_ts", "interval_indptr" ],
                dtype=np.int64,
            )

        for idx, log_file in enumerate(self._log_files):
            log.info(f"Processing {log_file}...")
            log_df = pd.read_csv(
                log_file,
                delimiter=",",
            )
            df_grouped_by_user = log_df.groupby(["user_id", "date"]).agg(list).reset_index()

            for event, weight in self._event_merge_weight.items():
                df_grouped_by_user[event] = df_grouped_by_user[event].apply(
                    lambda seq: np.where(np.array(seq) == 0, 0, weight)
                )

            events = list(self._event_merge_weight.keys())
            df_grouped_by_user["action_weights"] = df_grouped_by_user.apply(
                lambda row: [int(sum(x)) for x in zip(*[row[col] for col in events])],
                axis=1,
            )
            df_grouped_by_user = df_grouped_by_user[["user_id", "date"] + seq_cols]

            # ts1 = time.time()
            df_sorted = df_grouped_by_user.apply(create_interval, axis=1)
            df_filtered = df_sorted.groupby(level=0).apply(
                lambda group: group.reset_index(drop=True).reindex(np.arange(np.int64(group.num_intervals))).fillna(method='ffill')
            )
            
            df_filtered["interval_counter"] = df_filtered.index.get_level_values(1) + 1
            df_filtered = df_filtered.apply(filter_interval2, axis=1)
            df_filtered.reset_index()
            # ts2 = time.time()

            # print(ts2 - ts1)

            if idx == 0:
                sequence_df = df_sorted
                batching_df = df_filtered
            else:
                sequence_df = pd.concat([sequence_df, df_sorted])
                batching_df = pd.concat([batching_df, df_filtered])

        log.info("Merging user features...")
        user_features_df = pd.read_csv(self._user_features_file, delimiter=",")

        contextual_feature_names = self._contextual_feature_names.copy()
        contextual_feature_names.remove("user_id")
        for col in contextual_feature_names:
            user_features_df[col] = _one_hot_encode(user_features_df[col])

        self._post_process_use_splits(
            user_features_df,
            sequence_df,
            batching_df,
            "user_id",
            contextual_feature_names=self._contextual_feature_names,
            item_feature_name="video_id",
            action_feature_name="action_weights",
            sequence_file=self._output_file,
            batching_file=self._batch_file,
        )


dataset_names = (
    "ml-1m",
    "ml-20m",
    "kuairand-pure",
    "kuairand-1k",
    "kuairand-27k",
)

def get_common_preprocessors(dataset_path: str):
    """
    Get common data preprocessors.

    Returns:
        dict: Dictionary of common data preprocessors. The valid keys are
        "ml-1m", "ml-20m", "kuairand-pure", "kuairand-1k", "kuairand-27k".
    """
    data_path = dataset_path if dataset_path else "tmp_data"
    ml_1m_dp = MovielensDataProcessor(
        "http://files.grouplens.org/datasets/movielens/ml-1m.zip",
        data_path=data_path,
        file_name="movielens1m.zip",
        prefix="ml-1m",
    )
    ml_20m_dp = MovielensDataProcessor(
        "http://files.grouplens.org/datasets/movielens/ml-20m.zip",
        data_path=data_path,
        file_name="movielens20m.zip",
        prefix="ml-20m",
    )
    kuairand_pure_dp = DLRMKuaiRandProcessor(
        download_url="https://zenodo.org/records/10439422/files/KuaiRand-Pure.tar.gz",
        data_path=data_path,
        file_name="KuaiRand-Pure.tar.gz",
        prefix="KuaiRand-Pure",
    )
    kuairand_1k_dp = DLRMKuaiRandProcessor(
        download_url="https://zenodo.org/records/10439422/files/KuaiRand-1K.tar.gz",
        data_path=data_path,
        file_name="KuaiRand-1K.tar.gz",
        prefix="KuaiRand-1K",
    )
    kuairand_27k_dp = DLRMKuaiRandProcessor(
        download_url="https://zenodo.org/records/10439422/files/KuaiRand-27K.tar.gz",
        data_path=data_path,
        file_name="KuaiRand-27K.tar.gz",
        prefix="KuaiRand-27K",
    )
    return {key: locals()[f"{key}_dp".replace('-', '_')] for key in dataset_names}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocessor")
    parser.add_argument(
        "--dataset_name", choices=list(dataset_names) + ["all"]
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
    )
    args = parser.parse_args()

    common_preprocessors = get_common_preprocessors(args.dataset_path)
    if args.dataset_name == "all":
        for dataset_name in common_preprocessors.keys():
            dp = common_preprocessors[dataset_name]
            dp.preprocess()
    else:
        dp = common_preprocessors[args.dataset_name]
        dp.preprocess()
