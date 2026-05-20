"""
api/schemas.py -- Pydantic models for request/response validation
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# System
# ---------------------------------------------------------------------------
class HealthResponse(BaseModel):
    status: str = "healthy"
    version: str = "1.0.0"
    timestamp: str


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
class QualityScores(BaseModel):
    completeness: Optional[float] = None
    consistency: Optional[float] = None
    accuracy: Optional[float] = None
    uniqueness: Optional[float] = None
    timeliness: Optional[float] = None
    validity: Optional[float] = None


class ValidateResponse(BaseModel):
    batch_id: str
    timestamp: str
    rows_processed: int
    columns: List[str]
    quality_scores: Dict[str, Any]
    overall_score: float
    passed: bool = Field(alias="pass")
    pass_threshold: float
    issues_count: int
    issues: List[Dict[str, Any]]
    audit_trail: List[Dict[str, Any]]
    processing_time_ms: int


# ---------------------------------------------------------------------------
# Profiling
# ---------------------------------------------------------------------------
class ProfileResponse(BaseModel):
    rows: int
    columns: int
    profile: Dict[str, Any]
    processing_time_ms: int


# ---------------------------------------------------------------------------
# Anomaly Detection
# ---------------------------------------------------------------------------
class AnomalyResponse(BaseModel):
    rows_analyzed: int
    anomalies_found: int
    anomalies: List[Dict[str, Any]]
    processing_time_ms: int


# ---------------------------------------------------------------------------
# Rules
# ---------------------------------------------------------------------------
class RulesResponse(BaseModel):
    columns_analyzed: int
    rules_generated: int
    rules: Dict[str, Any]
    processing_time_ms: int


# ---------------------------------------------------------------------------
# S3 Integration
# ---------------------------------------------------------------------------
class S3ValidateRequest(BaseModel):
    bucket: str = Field(..., description="S3 bucket name")
    key: str = Field(..., description="S3 object key (file path)")
    region: str = Field("us-east-1", description="AWS region")
    reference_key: Optional[str] = Field(None, description="S3 key for reference data")
    rules_key: Optional[str] = Field(None, description="S3 key for rules JSON file")
    pass_threshold: float = Field(85.0, description="Quality score pass threshold")
    output_bucket: Optional[str] = Field(None, description="S3 bucket for saving results")
    output_prefix: Optional[str] = Field("reports", description="S3 prefix for output files")


# ---------------------------------------------------------------------------
# AI Investigation
# ---------------------------------------------------------------------------
class InvestigateRequest(BaseModel):
    bucket: str = Field(..., description="S3 bucket containing the data")
    key: str = Field(..., description="S3 object key")
    region: str = Field("us-east-1", description="AWS region")


class InvestigateResponse(BaseModel):
    batch_id: str
    timestamp: str
    source: str
    rows_processed: int
    overall_score: float
    passed: bool = Field(alias="pass")
    issues_count: int
    top_problem_columns: List[Dict[str, Any]]
    severity_breakdown: Dict[str, int]
    recommendation: str
    audit_trail: List[Dict[str, Any]]
    processing_time_ms: int


# ---------------------------------------------------------------------------
# Batch History
# ---------------------------------------------------------------------------
class BatchSummary(BaseModel):
    batch_id: str
    timestamp: str
    rows_processed: Optional[int] = None
    overall_score: Optional[float] = None
    issues_count: Optional[int] = None
    passed: Optional[bool] = None


class BatchListResponse(BaseModel):
    batches: List[Dict[str, Any]]
    count: int


class BatchDetailResponse(BaseModel):
    batch: Dict[str, Any]
    issues: List[Dict[str, Any]]
    issues_count: int
    audit_trail: List[Dict[str, Any]]
