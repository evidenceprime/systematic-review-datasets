# coding=utf-8
# Copyright 2024
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import math
import os
from typing import List, Tuple, Dict, Any, Optional

import datasets
import csmed

from csmed.loader.bigbiohub import BigBioConfig
from csmed.loader.bigbiohub import Tasks
from csmed.loader.bigbiohub import text_features

_LANGUAGES = ["English"]
_PUBMED = True
_LOCAL = False

_CITATION = """\
"""

_DATASETNAME = "merged_tar_sigir_srupdates"
_DISPLAYNAME = "merged_tar_sigir_srupdates"

_DESCRIPTION = """\
Merged dataset that combines tar2017, tar2018, tar2019, sigir2017, and sr_updates
with review-level deduplication across sources.
"""

_HOMEPAGE = "https://github.com/WojciechKusa/systematic-review-datasets"
_LICENSE = "Unknown"

_SUPPORTED_TASKS = [Tasks.TEXT_CLASSIFICATION, Tasks.QUESTION_ANSWERING]

_SOURCE_VERSION = "1.0.0"
_BIGBIO_VERSION = "1.0.0"

_CLASS_NAMES = ["included", "excluded"]

def _get_csmed_root() -> str:
    env_root = os.environ.get("CSMED_ROOT")
    if env_root:
        return env_root
    return os.path.dirname(os.path.abspath(csmed.__file__))


_CSMED_ROOT = _get_csmed_root()

_DATASET_SOURCES = [
    {
        "dataset_id": "tar2019",
        "path": os.path.join(_CSMED_ROOT, "datasets", "tar2019", "tar2019.py"),
        "config": "tar2019_all_source",
        "review_field": "review_name",
    },
    {
        "dataset_id": "tar2018",
        "path": os.path.join(_CSMED_ROOT, "datasets", "tar2018", "tar2018.py"),
        "config": "tar2018_all_source",
        "review_field": "review_name",
    },
    {
        "dataset_id": "tar2017",
        "path": os.path.join(_CSMED_ROOT, "datasets", "tar2017", "tar2017.py"),
        "config": "tar2017_all_source",
        "review_field": "review_name",
    },
    {
        "dataset_id": "sr_updates",
        "path": os.path.join(_CSMED_ROOT, "datasets", "sr_updates", "sr_updates.py"),
        "config": "sr_updates_all_source",
        "review_field": "review_id",
    },
    {
        "dataset_id": "sigir2017",
        "path": os.path.join(_CSMED_ROOT, "datasets", "sigir2017", "sigir2017.py"),
        "config": "sigir2017_all_source",
        "review_field": "review_name",
    },
]


def _load_dataset(path: str, name: str) -> datasets.Dataset:
    dataset = datasets.load_dataset(path, name=name, trust_remote_code=True)
    if isinstance(dataset, datasets.DatasetDict):
        splits = [dataset[split] for split in dataset.keys()]
        if len(splits) == 1:
            return splits[0]
        return datasets.concatenate_datasets(splits)
    return dataset


def _to_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    return str(value)


def _coerce_label(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _normalize_example(
    dataset_id: str, example: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    if dataset_id == "sr_updates":
        review_name = _to_str(example.get("review_id"))
    else:
        review_name = _to_str(example.get("review_name"))

    label = _coerce_label(example.get("label"))
    if label is None:
        return None

    return {
        "review_name": review_name,
        "pmid": _to_str(example.get("pmid")),
        "title": example.get("title"),
        "abstract": example.get("abstract"),
        "label": label,
    }


def _make_text(title: Optional[str], abstract: Optional[str]) -> str:
    parts = [part for part in [title, abstract] if part]
    return "\n\n".join(parts)


class MergedTarSigirSrUpdatesDataset(datasets.GeneratorBasedBuilder):
    """Merged TAR/SIGIR/SR-Updates dataset with review-level de-duplication."""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    BIGBIO_VERSION = datasets.Version(_BIGBIO_VERSION)

    BUILDER_CONFIGS = []
    dataset_versions = ["all"]
    for dataset_version in dataset_versions:
        BUILDER_CONFIGS.append(
            BigBioConfig(
                name=f"merged_tar_sigir_srupdates_{dataset_version}_source",
                version=SOURCE_VERSION,
                description=(
                    "merged tar2017/2018/2019, sigir2017, sr_updates source schema"
                ),
                schema="source",
                subset_id=f"merged_tar_sigir_srupdates_{dataset_version}",
            )
        )
        BUILDER_CONFIGS.append(
            BigBioConfig(
                name=f"merged_tar_sigir_srupdates_{dataset_version}_bigbio_text",
                version=BIGBIO_VERSION,
                description=(
                    "merged tar2017/2018/2019, sigir2017, sr_updates BigBio schema"
                ),
                schema="bigbio_text",
                subset_id=f"merged_tar_sigir_srupdates_{dataset_version}",
            )
        )

    DEFAULT_CONFIG_NAME = "merged_tar_sigir_srupdates_all_source"

    def _info(self) -> datasets.DatasetInfo:
        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "review_name": datasets.Value("string"),
                    "pmid": datasets.Value("string"),
                    "title": datasets.Value("string"),
                    "abstract": datasets.Value("string"),
                    "label": datasets.ClassLabel(names=_CLASS_NAMES),
                }
            )
        elif self.config.schema == "bigbio_text":
            features = text_features
        else:
            raise ValueError(f"Unsupported schema {self.config.schema}")

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager) -> List[datasets.SplitGenerator]:
        """Returns SplitGenerators."""
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={},
            )
        ]

    def _generate_examples(self) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""
        seen_reviews: set[str] = set()
        uid = 0

        for source in _DATASET_SOURCES:
            dataset = _load_dataset(source["path"], source["config"])
            review_field = source["review_field"]
            review_names = {
                _to_str(example.get(review_field)) for example in dataset
            }
            review_names.discard(None)
            allowed_reviews = review_names - seen_reviews
            if not allowed_reviews:
                continue

            for example in dataset:
                review_name = _to_str(example.get(review_field))
                if review_name not in allowed_reviews:
                    continue
                normalized = _normalize_example(source["dataset_id"], example)
                if normalized is None:
                    continue

                uid += 1
                if self.config.schema == "source":
                    yield str(uid), normalized
                elif self.config.schema == "bigbio_text":
                    text = _make_text(normalized["title"], normalized["abstract"])
                    data = {
                        "id": str(uid),
                        "document_id": normalized["pmid"],
                        "text": text,
                        "labels": [normalized["label"]],
                    }
                    yield str(uid), data

            seen_reviews.update(allowed_reviews)


if __name__ == "__main__":
    x = datasets.load_dataset(__file__, name="merged_tar_sigir_srupdates_all_source")
    print(type(x))
    print(x)
