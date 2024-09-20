import threading
import time
from datetime import datetime, timezone

import schedule
from injector import inject

from businesses.training_business import TrainingBusiness
from common.enums.job_type import JobType
from infrastructure.repositories.job_repository import JobRepository


class JobBusiness:
    @inject
    def __init__(self, training_business: TrainingBusiness, job_repository: JobRepository):
        self.job_repository = job_repository
        self.training_business = training_business

    def training_job(self, training_id: int = None):
        print("Start background job")

        in_progress_jobs = self.job_repository.get_inprogress_job_by_job_type(JobType.Training)

        if in_progress_jobs:
            print("There is another in progress job!")
            return

        job_id = self.job_repository.insert(JobType.Training, datetime.now(timezone.utc))

        # try:
        self.training_business.run_trainings(training_id)
        # except Exception as e:
        #     print(f"An error occurred during training: {e}")
        # finally:
        self.job_repository.update(job_id, datetime.now(timezone.utc))
        print("Finished background job")

    def run_scheduler(self):
        schedule.every(1).hour.do(self.training_job)

        scheduler_thread = threading.Thread(target=self._run_scheduler)
        scheduler_thread.daemon = True
        scheduler_thread.start()

        threading.Thread(target=self._delay_first_run).start()

    def _run_scheduler(self):
        while True:
            schedule.run_pending()
            time.sleep(1)

    def _delay_first_run(self):
        pass
