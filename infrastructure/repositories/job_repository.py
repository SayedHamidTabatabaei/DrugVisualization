from datetime import datetime
from typing import Optional

from common.enums.job_type import JobType
from core.domain.job import Job
from infrastructure.mysqldb.mysql_repository import MySqlRepository


class JobRepository(MySqlRepository):
    def __init__(self):
        super().__init__('jobs')

    def insert(self, job_type: JobType, start: datetime) -> int:
        data = Job(job_type=job_type, start=start)

        id = super().insert(data)

        return id

    def update(self, id: int, end: datetime):
        job = self.get_job_by_id(id)

        job.end = end

        update_columns = ['end']

        rowcount = super().update(job, update_columns)

        return rowcount

    def delete(self, id: int):

        rowcount = super().delete(id)

        return rowcount

    def get_job_by_id(self, id) -> Job:
        result, _ = self.call_procedure('GetJobById', [id])

        id, job_type, start, end = result[0][0]
        return Job(id=id, job_type=job_type, start=start, end=end)

    def get_inprogress_job_by_job_type(self, job_type) -> Optional[Job]:
        result, _ = self.call_procedure('GetInProgressJobByJobType', [job_type.value])

        if not result or not result[0]:
            return None

        id, job_type, start, end = result[0][0]
        return Job(id=id, job_type=job_type, start=start, end=end)

    def incorrect_job_delete(self):
        self.call_procedure('DeleteIncorrectJobs')
