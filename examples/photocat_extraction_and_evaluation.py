import sys
sys.path.append("chemdataextractor")
sys.path.append("cdedev")
sys.path.append("CDEDatabase")
sys.path.append(".")

from e2e_workflow.extraction.extractor import CDEDatabaseExtractor
from bert_paper.photocatalyst_models_nturn_qa import PhotocatalyticActivity, PhotocatalyticEfficiency
from photocatalyst_models import filter_results, is_valid_document, adjascent_sections
from chemdataextractor.doc.text import Sentence, Citation, Footnote
from chemdataextractor.nlp.subsentence import NoneSubsentenceExtractor
from photocatalyst_models import ApparentQuantumYield, HydrogenEvolution, HydrogenEvolution2, HydrogenEvolution3, Additive
from chemdataextractor.model.units import Dimensionless
from chemdataextractor.model import ModelList
from chemdataextractor.parse.quantity import extract_units
from e2e_workflow.evaluation.compare_databases import Comparer
import datetime
import os
import inspect

DISABLE_SUBSENTENCES = True
SHOULD_REMOVE_SUBRECORDS_IF_USED = True

document_dir = os.getenv("DOCUMENT_DIR")
save_root_dir = os.getenv("OUTPUT_DIR")
annotated_dir = os.getenv("ANNOTATED_DIR")
wandb_project = os.getenv("WANDB_PROJECT")
cache_dir = document_dir + "_cache"
use_mpi = bool(int(os.getenv("USE_MPI"))) if os.getenv("USE_MPI") is not None else True
use_wandb = bool(int(os.getenv("CDE_USE_WANDB"))) if os.getenv("CDE_USE_WANDB") is not None else True

wandb_config = None
if use_wandb:
    wandb_config = {
        "save_root_dir": save_root_dir,
        "use_mpi": True,
        "model_name": os.getenv("BERT_MODEL_NAME")
    }

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
    use_wandb=use_wandb,
    wandb_config=wandb_config,
    wandb_project=wandb_project,
    wandb_run_name=os.path.basename(save_root_dir),
    wandb_save_files=[(inspect.getmodule(PhotocatalyticActivity).__file__)]
)

all_start_time = datetime.datetime.now()

extractor.extract(document_dir)

print(f"Extraction as a whole took: {datetime.datetime.now() - all_start_time}")


def filter_records_for(dimensions_list):
    def filter_records_internal(records):
        dimensions_combined = Dimensionless()
        for dimension in dimensions_list:
            dimensions_combined *= dimension

        filtered_records = ModelList()
        for record in records:
            record_units = extract_units(record.raw_units, dimensions_combined)
            if record_units is not None and record_units.dimensions in dimensions_list:
                filtered_records.append(record)
        return filtered_records
    return filter_records_internal


lax_fields = [
    "compound.names",
    "cocatalyst.compound.names",
    "additive.compound.names"
]
ignore_fields = [
    "value", "units", "error",
     "light_source_wavelength.units",
     "light_source_wavelength.value",
     "light_source_power.units",
     "light_source_power.value", "irradiation_time.units",
     "irradiation_time.value",
     "no_cocatalyst",
]
global_ignore_fields = [
    "error", "specifier", "labels",
    "roles", "is_reported_value", "raw_value",
    "raw_units", "extracted_value", "light_source",
    "irradiation_time"
]
models = {
    PhotocatalyticActivity: "val_units",
    PhotocatalyticEfficiency: "val_units",
}
primary_key_overrides = {
    Additive: "compound.names"
}

filter_func = filter_records_for([
    ApparentQuantumYield.dimensions,
    HydrogenEvolution.dimensions,
    HydrogenEvolution2.dimensions,
    HydrogenEvolution3.dimensions
])

comparer = Comparer(
    models,
    lax_fields,
    ignore_fields,
    global_ignore_fields,
    filter_func,
    primary_key_overrides
)

if not use_mpi:
    comparer.compare_and_log_to_wandb(
        [save_root_dir],
        [annotated_dir],
    )
else:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    is_main_thread = (rank == 0)
    if is_main_thread:
        print("IS MAIN THREAD AND COMPARING")
        comparer.compare_and_log_to_wandb(
            [save_root_dir],
            [annotated_dir],
        )
