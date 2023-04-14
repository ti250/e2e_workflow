from bert_paper.photocatalyst_models_nturn_qa import PhotocatalyticActivity, PhotocatalyticEfficiency
from photocatalyst_models import ApparentQuantumYield, HydrogenEvolution, HydrogenEvolution2, HydrogenEvolution3, Additive
from chemdataextractor.model.units import Dimensionless
from chemdataextractor.model import ModelList
from chemdataextractor.parse.quantity import extract_units
from e2e_workflow.evaluation.compare_databases import Comparer


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
    "raw_units", "extracted_value"
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

comparer.compare_from_terminal_args()
