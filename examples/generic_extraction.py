from ..extractor import CDEDatabaseExtractor
from allennlp.data.token_indexers import PretrainedBertIndexer
from genericextractor.lemma_tagger import LemmaTagger

from bert_paper.photocatalyst_models_nturn_qa import PhotocatalyticActivity, PhotocatalyticEfficiency
from photocatalyst_models import filter_results, is_valid_document, adjascent_sections
from chemdataextractor.doc.text import Sentence, Citation, Footnote
from chemdataextractor.nlp.subsentence import NoneSubsentenceExtractor
from chemdataextractor.nlp.allennlpwrapper import _AllenNlpTokenTagger, ProcessedTextTagger, AllenNlpWrapperTagger
from chemdataextractor.data import Package, PACKAGES, find_data
import datetime
import os
from genericextractor.generic_extractor import GENERIC_EXTRACTOR_LABEL_TYPE

from bert_paper.n_turn_ge import BertExtractedGenericModel

BUILD_NUM = os.getenv("BUILD_NUM")
SHOULD_REMOVE_SUBRECORDS_IF_USED = True
DISABLE_SUBSENTENCES = True

document_dir = "datasets/processed/photocat_train"
save_root_dir = f"photocat_datasets/cde_run/{BUILD_NUM}"
cache_dir = document_dir + "_cache"

if DISABLE_SUBSENTENCES:
    Sentence.subsentence_extractor = NoneSubsentenceExtractor()

tag_type = GENERIC_EXTRACTOR_LABEL_TYPE
matscholar_archive_package = Package(
    "models/bert_matscholar",
    remote_path="https://cdemodelsstorage.blob.core.windows.net/cdemodels/bert_matscholar.tar.gz",
    untar=True
)
# Append to PACKAGES so that find_data works
PACKAGES.append(matscholar_archive_package)

indexers = {
    "bert": PretrainedBertIndexer(
        do_lowercase=False,
        use_starting_offsets=True,
        truncate_long_sequences=False,
        pretrained_model=find_data("models/scibert_cased_vocab-1.0.txt")
    ),
}


class MatscholarTagger(AllenNlpWrapperTagger):
    overrides = {"model.text_field_embedder.token_embedders.bert.pretrained_model": find_data("models/scibert_cased_weights-1.0.tar.gz")}


allenwrappertagger = MatscholarTagger(
    tag_type=tag_type,
    archive_location=find_data("models/bert_matscholar"),
    indexers=indexers
)


Sentence.taggers.extend([allenwrappertagger, _AllenNlpTokenTagger(), ProcessedTextTagger(), LemmaTagger()])


extractor = CDEDatabaseExtractor(
    models=[BertExtractedGenericModel],
    save_root_dir=save_root_dir,
    document_args={
        "adjascent_sections_for_merging": adjascent_sections,
        "skip_elements": [Citation, Footnote],
        "_should_remove_subrecord_if_merged_in": SHOULD_REMOVE_SUBRECORDS_IF_USED
    },
    filter_results=None,
    is_valid_document=None,
    cache_dir=cache_dir,
)

all_start_time = datetime.datetime.now()

extractor.extract(document_dir)

print(f"Extraction as a whole took: {datetime.datetime.now() - all_start_time}")
