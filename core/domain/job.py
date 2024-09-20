from sqlalchemy import Column, Enum as SqlEnum, DateTime

from common.enums.job_type import JobType
from core.domain.base_model import BaseModel


class Job(BaseModel):
    __tablename__ = 'jobs'

    job_type = Column(SqlEnum(JobType), nullable=False)
    start = Column(DateTime, nullable=False)
    end = Column(DateTime, nullable=True)
