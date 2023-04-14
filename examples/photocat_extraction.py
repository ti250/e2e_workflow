from e2e_workflow.extraction.extractor import CDEDatabaseExtractor
from bert_paper.photocatalyst_models_nturn_qa import PhotocatalyticActivity, PhotocatalyticEfficiency
from photocatalyst_models import filter_results, is_valid_document, adjascent_sections
from chemdataextractor.doc.text import Sentence, Citation, Footnote
from chemdataextractor.nlp.subsentence import NoneSubsentenceExtractor
import datetime
import os

DISABLE_SUBSENTENCES = True
SHOULD_REMOVE_SUBRECORDS_IF_USED = True

document_dir = os.getenv("DOCUMENT_DIR")
save_root_dir = os.getenv("OUTPUT_DIR")
cache_dir = document_dir + "_cache"
use_mpi = bool(int(os.getenv("USE_MPI"))) if os.getenv("USE_MPI") is not None else True

if DISABLE_SUBSENTENCES:
    Sentence.subsentence_extractor = NoneSubsentenceExtractor()

extractor = CDEDatabaseExtractor(
    models=[PhotocatalyticActivity, PhotocatalyticEfficiency],
    save_root_dir=save_root_dir,
    document_args={
        "adjascent_sections_for_merging": adjascent_sections,
        "skip_elements": [Citation, Footnote],
        "_should_remove_subrecord_if_merged_in": SHOULD_REMOVE_SUBRECORDS_IF_USED
    },
    filter_results=filter_results,
    is_valid_document=is_valid_document,
    cache_dir=cache_dir,
    use_mpi=use_mpi,
)

all_start_time = datetime.datetime.now()

extractor.extract(document_dir)

print(f"Extraction as a whole took: {datetime.datetime.now() - all_start_time}")
