from e2e_workflow.extraction.base_extractor import BaseExtractor
from cdedatabase import CDEDatabase, JSONCoder
import wandb

import os


class CDEDatabaseExtractor(BaseExtractor):
    def __init__(
        self,
        models,
        save_root_dir,
        document_args=None,
        filter_results=None,
        is_valid_document=None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.models = models
        self.save_root_dir = save_root_dir

        self.document_args = {}
        if document_args is not None:
            self.document_args = document_args

        self.filter_results = filter_results
        self.is_valid_document = is_valid_document

    def will_start_extraction(self):
        super().will_start_extraction()
        if not os.path.isdir(self.save_root_dir):
            os.mkdir(self.save_root_dir)

    def should_open_file(self, filename):
        db_name = self.db_name_for_file(filename)
        should_open = not os.path.exists(db_name)
        if not should_open:
            print(f"Skipping {filename} as already exists")
        return should_open

    def should_process_document(self, document):
        if self.is_valid_document is None:
            return True

        is_valid = self.is_valid_document(document)

        return is_valid

    def configure_document(self, document):
        super().configure_document(document)

        document.models = self.models

        for key, value in self.document_args.items():
            setattr(document, key, value)

    def postprocess_records(self, records, filename):
        if self.filter_results is not None and len(records):
            records = self.filter_results(records)
        db_name = self.db_name_for_file(filename)
        db = CDEDatabase(db_name, coder=JSONCoder())
        db.write(records)

    def db_name_for_file(self, filename):
        only_filename = os.path.split(filename)[-1]
        no_ext_filename = os.path.splitext(only_filename)[0]
        return os.path.join(self.save_root_dir, no_ext_filename)
