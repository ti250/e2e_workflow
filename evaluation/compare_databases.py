from cdedatabase import CDEDatabase, JSONCoder
from cdedatabase.results import Results
from photocatalyst_models import ApparentQuantumYield, SolarToHydrogen, HydrogenEvolution, HydrogenEvolution2, HydrogenEvolution3, Additive
from e2e_workflow.evaluation.compare_records import compare_records, Statistics
from pprint import pprint
import sys
import os
import copy
import wandb


def papers_list(directory):
    return [folder for folder in os.listdir(directory) if os.path.isdir(os.path.join(directory, folder))]


class Comparer:
    def __init__(
        self,
        models_and_primary_keys,
        lax_fields=None,
        ignore_fields=None,
        global_ignore_fields=None,
        postprocess_filter=None,
        primary_key_overrides=None,
    ):
        self.models_and_primary_keys = models_and_primary_keys
        self.postprocess_filter = postprocess_filter

        self.lax_fields = lax_fields
        if lax_fields is None:
            self.lax_fields = []

        self.ignore_fields = ignore_fields
        if ignore_fields is None:
            self.ignore_fields = []

        self.global_ignore_fields = global_ignore_fields
        if global_ignore_fields is None:
            self.global_ignore_fields = []

        self.primary_key_overrides = primary_key_overrides
        if primary_key_overrides is None:
            self.primary_key_overrides = {}

    def compare(
        self,
        verbose,
        show_all,
        is_lax,
        extracted_db_dirs,
        annotated_db_dirs,
        _log_to_wandb=False
    ):
        all_models_stats = Statistics(0, 0, 0, 0)
        cross_model_stats = {}
        lax_fields = []

        if is_lax:
            lax_fields = self.lax_fields

        paper_names_with_db_index = []
        for index, annotated_db_dir in enumerate(annotated_db_dirs):
            paper_names = papers_list(annotated_db_dir)
            new_entries = [(paper_name, index) for paper_name in paper_names]
            paper_names_with_db_index.extend(new_entries)

        for (model, primary_key) in self.models_and_primary_keys.items():
            primary_keys = {"default": primary_key}
            primary_keys.update(self.primary_key_overrides)
            overall_stats = None
            print('\n\n\n')
            print(f'Records for {model.__name__} \n====================================================')
            for paper_name, db_index in paper_names_with_db_index:
                annotated_db_dir = annotated_db_dirs[db_index]
                extracted_db_dir = extracted_db_dirs[db_index]
                annotated_database = CDEDatabase(
                    os.path.join(annotated_db_dir, paper_name),
                    coder=JSONCoder()
                )
                extracted_database = CDEDatabase(
                    os.path.join(extracted_db_dir, paper_name),
                    coder=JSONCoder()
                )
                records_correct = annotated_database.records(model).all()
                records_found = extracted_database.records(model).all()
                if self.postprocess_filter is not None:
                    records_found = self.postprocess_filter(records_found)

                stats = compare_records(
                    records_correct,
                    records_found,
                    primary_keys=primary_keys,
                    verbose=verbose,
                    ignore_fields=self.ignore_fields,
                    global_ignore_fields=self.global_ignore_fields,
                    lax_fields=lax_fields
                )
                paper_stat = Statistics(0, 0, 0, 0)

                if stats is not None:
                    if overall_stats is not None:
                        for key, value in overall_stats.items():
                            overall_stats[key] = overall_stats[key] + stats[key]
                    else:
                        overall_stats = stats
                    for key, stat in stats.items():
                        paper_stat += stat

                if ((paper_stat.tp != 0 or paper_stat.fp != 0 or paper_stat.fn != 0) and verbose) or show_all:
                    print("--------------------------------")
                    print(paper_name)
                    print(paper_stat)
                    print("  -Precision:", paper_stat.precision())
                    print("  -Recall:", paper_stat.recall())
                    print("\n\n\n")
            print("FINISHED RECORDS FOR MODEL")

            cumulative_stats = Statistics(0, 0, 0, 0)
            print("\n--------------------------------------------")
            pprint(overall_stats)
            print("--------------------------------------------\n")
            if overall_stats is not None:
                for field, stat in overall_stats.items():
                    cumulative_stats += stat
                    print(field)
                    print("  -Precision:", stat.precision())
                    print("  -Recall:", stat.recall())
                    if field in cross_model_stats:
                        cross_model_stats[field] += stat
                    else:
                        cross_model_stats[field] = stat

            print(cumulative_stats)
            print("  -Precision:", cumulative_stats.precision())
            print("  -Recall:", cumulative_stats.recall())

            all_models_stats += cumulative_stats


        print("\n\n====================================================\nALL MODELS:")
        print(all_models_stats)
        print("  -Precision:", all_models_stats.precision())
        print("  -Recall:", all_models_stats.recall())
        print("  -F1:", all_models_stats.f1())

        print("\n----------------------------------------------------\nBREAKDOWN:")
        for field, stat in cross_model_stats.items():
            print(field)
            print("  ", stat)
            print("  -Precision:", stat.precision())
            print("  -Recall:", stat.recall())
            print("  -F1:", stat.f1())

        if _log_to_wandb:
            lax_string = "lax-" if is_lax else ""
            stats_dict = {
                f"all_models/{lax_string}precision": all_models_stats.precision(),
                f"all_models/{lax_string}recall": all_models_stats.recall(),
                f"all_models/{lax_string}f1": all_models_stats.f1()
            }

            for field, stat in cross_model_stats.items():
                stats_dict.update({
                    f"{field}/{lax_string}precision": stat.precision(),
                    f"{field}/{lax_string}recall": stat.recall(),
                    f"{field}/{lax_string}f1": stat.f1()
                })

            print("STATS DICT: ", stats_dict)
            wandb.log(stats_dict)

    def compare_from_terminal_args(self):
        show_all = False
        verbose = False
        is_lax = False
        extracted_db_dirs = []
        annotated_db_dirs = []
        if len(sys.argv):
            args = copy.copy(sys.argv)[1:]

            if "--verbose" in args:
                verbose = True
                args.remove("--verbose")

            if "--show_all" in args:
                show_all = True
                args.remove("--show_all")

            if "--lax" in args:
                is_lax = True
                args.remove("--lax")

            if len(args):
                extracted_db_dirs = args[0].split(",")
                annotated_db_dirs = args[1].split(",")
                assert len(annotated_db_dirs) == len(extracted_db_dirs)

        self.compare(
            verbose,
            show_all,
            is_lax,
            extracted_db_dirs,
            annotated_db_dirs
        )

    def compare_and_log_to_wandb(self, extracted_db_dirs, annotated_db_dirs):
        print("COMPARING LAX")
        self.compare(
            verbose=True,
            show_all=False,
            is_lax=True,
            extracted_db_dirs=extracted_db_dirs,
            annotated_db_dirs=annotated_db_dirs,
            _log_to_wandb=True
        )

        print("COMPARING NO LAX")
        self.compare(
            verbose=True,
            show_all=False,
            is_lax=False,
            extracted_db_dirs=extracted_db_dirs,
            annotated_db_dirs=annotated_db_dirs,
            _log_to_wandb=True
        )

