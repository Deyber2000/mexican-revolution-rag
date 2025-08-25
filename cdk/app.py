#!/usr/bin/env python3
"""
CDK App for Mexican Revolution RAG System
Deploys the RAG system to AWS with production-ready infrastructure
"""

import aws_cdk as cdk
from rag_stack import RAGStack

app = cdk.App()

# Create the RAG stack
RAGStack(
    app,
    "MexicanRevolutionRAGStack",
    env=cdk.Environment(
        account=app.node.try_get_context("account"),
        region=app.node.try_get_context("region") or "us-east-1",
    ),
    description="Production RAG system for Mexican Revolution Q&A",
)

app.synth()
