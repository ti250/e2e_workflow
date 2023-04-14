from cdedatabase import CDEDatabase, JSONCoder
from photocatalyst_models import filter_results, is_valid_document, adjascent_sections

from chemdataextractor import Document
from chemdataextractor.doc.text import Citation, Footnote
from chemdataextractor.model.base import ModelList
from chemdataextractor.doc.document_cacher import PlainTextCacher

import wandb
import datetime
import os
from pprint import pprint

print(wandb.__path__)

AWAITING_DATA_TAG = 111
RETURNING_DATA_TAG = 222
# is_main_thread = (rank == 0)


class BaseExtractor:
    def __init__(self, cache_dir=None, use_mpi=True, use_wandb=False, wandb_project=None, wandb_config=None, wandb_run_name=None):
        self.cache_dir = cache_dir
        if self.cache_dir is not None:
            self.cacher = PlainTextCacher(cache_dir)

        self.use_mpi = use_mpi
        if use_mpi:
            from mpi4py import MPI
            self.comm = MPI.COMM_WORLD
            self.rank = self.comm.Get_rank()
            self.size = self.comm.Get_size()
            self.is_main_thread = (self.rank == 0)

        self.use_wandb = use_wandb
        self.wandb_project = wandb_project
        self.wandb_config = wandb_config
        self.wandb_run_name = wandb_run_name

    def will_start_extraction(self):
        if self.use_wandb:
            if wandb.run is None:
                self.create_wandb_run()
            else:
                print("EXISTING RUN", wandb.run)

    def create_wandb_run(self):
        print("WANDB CONFIG:", self.wandb_config)
        return wandb.init(
            project=self.wandb_project,
            config=self.wandb_config,
            name=self.wandb_run_name
        )

    def should_open_file(self, file):
        return True

    def should_process_document(self, document):
        return True

    def configure_document(self, document):
        pass

    def postprocess_records(self, records, filename):
        pass

    def extract_paper(self, document_path):
        doc_start_time = datetime.datetime.now()

        if self.should_open_file(document_path):
            did_use_cache = False
            doc = Document.from_file(document_path)
            self.configure_document(doc)
            if self.cache_dir is not None:
                try:
                    self.cacher.hydrate_document(doc, document_path)
                    did_use_cache = True
                except AttributeError:
                    pass

            document_records = ModelList()
            if self.should_process_document(doc):
                document_records = doc.records
            else:
                print(f"CANCELLED DOCUMENT {document_path}")

            self.postprocess_records(document_records, document_path)

            if not did_use_cache and self.cache_dir is not None:
                self.cacher.cache_document(doc, document_path, overwrite_cache=True)

        doc_end_time = datetime.datetime.now()

        print(f"{document_path} took:", doc_end_time - doc_start_time)

    def _extract_single_threaded(self, document_dir, num_papers=None):

        all_start_time = datetime.datetime.now()

        filenames = [filename for filename in os.listdir(document_dir) if filename[0] != '.' and 'records.txt' not in filename]

        for index, filename in enumerate(filenames):
            if num_papers is not None and index > num_papers:
                break

            print(f"\n\n\nPaper {index + 1}/{len(filenames)}: {filename}")
            wandb.log({"num_papers_processed": index})
            full_path = os.path.join(document_dir, filename)

            self.extract_paper(full_path)

        print(f"Extraction as a whole took: {datetime.datetime.now() - all_start_time}")

    def _extract_mpi(self, document_dir, num_papers=None):
        from mpi4py import MPI

        if self.is_main_thread:
            all_start_time = datetime.datetime.now()
            index = 0
            n_finished = 0
            filenames = [filename for filename in os.listdir(document_dir) if filename[0] != '.' and 'records.txt' not in filename]
            num_papers = len(filenames) if num_papers is None else num_papers

            while True:
                status = MPI.Status()
                if index == 0:
                    for i in range(1, self.size):
                        filename = filenames[index]
                        full_path = os.path.join(document_dir, filename)
                        self._send_to_worker(i, full_path)
                        index += 1
                else:
                    _ = self.comm.recv(
                        source=MPI.ANY_SOURCE,
                        tag=MPI.ANY_TAG,
                        status=status
                    )
                    finished_worker_index = status.Get_source()
                    n_finished += 1
                    wandb.log({"num_papers_processed": n_finished})

                    if n_finished == index and index >= num_papers:
                        break
                    elif index < num_papers:
                        print(index, num_papers)
                        filename = filenames[index]
                        full_path = os.path.join(document_dir, filename)
                        self._send_to_worker(finished_worker_index, full_path)
                        index += 1
            for i in range(1, self.size):
                self._exit_worker(i)
            print(f"Extraction as a whole took: {datetime.datetime.now() - all_start_time}")

        else:
            self._start_worker()

    def _send_to_worker(self, worker_index, document_path):
        data = {
            "exit": False,
            "document_path": document_path
        }
        self.comm.send(data, dest=worker_index, tag=AWAITING_DATA_TAG)

    def _exit_worker(self, worker_index):
        data = {"exit": True}
        self.comm.send(data, dest=worker_index, tag=AWAITING_DATA_TAG)

    def _start_worker(self):
        while True:
            data = self.comm.recv(source=0, tag=AWAITING_DATA_TAG)
            if data["exit"]:
                break
            else:
                document_path = data["document_path"]
                try:
                    result = self.extract_paper(document_path)
                    self.comm.send(result, dest=0, tag=RETURNING_DATA_TAG)
                except Exception as e:
                    print(f"EXITED FOR {document_path} DUE TO: {e}")
                    self.comm.send([], dest=0, tag=RETURNING_DATA_TAG)
                    # break

    def extract(self, document_dir, num_papers=None):
        if not self.use_mpi or self.is_main_thread:
            self.will_start_extraction()

        if not self.use_mpi or self.size == 1:
            self._extract_single_threaded(document_dir, num_papers)
        else:
            self._extract_mpi(document_dir, num_papers)
