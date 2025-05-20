#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Question answering module using transformers.
"""

from typing import Dict, List, Tuple, Union, Any
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline


class QuestionAnswerer:
    """
    A class for answering questions using a QA model.
    """
    
    def __init__(self, modelName: str = "deepset/roberta-base-squad2") -> None:
        """
        Initialize the QuestionAnswerer.
        
        Args:
            modelName: Name of the QA model to use
        """
        self.modelName = modelName
        self.tokenizer = AutoTokenizer.from_pretrained(modelName)
        self.model = AutoModelForQuestionAnswering.from_pretrained(modelName)
        
        # Create a QA pipeline
        self.qa_pipeline = pipeline(
            "question-answering",
            model=self.model,
            tokenizer=self.tokenizer
        )
    
    def answerQuestion(self, 
                      question: str, 
                      context: Union[str, List[str]],
                      topK: int = 1) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Answer a question using the QA model.
        
        Args:
            question: The question to answer
            context: The context(s) to use for answering
            topK: Number of answers to return if multiple contexts
            
        Returns:
            Dictionary with answer, score, and context
            or list of such dictionaries if multiple contexts
        """
        if isinstance(context, list):
            # Join contexts into a single string with separators
            combined_context = " ".join(context)
            if not combined_context.strip():
                return {"answer": "No context provided to answer the question.", "score": 0.0, "context": ""}
            
            try:
                result = self.qa_pipeline(
                    question=question,
                    context=combined_context,
                    top_k=topK,
                    max_answer_len=512,
                    handle_impossible_answer=True
                )
                
                if isinstance(result, list):
                    for r in result:
                        r["context"] = combined_context
                    return result
                else:
                    result["context"] = combined_context
                    return result
            except Exception as e:
                # Fallback for errors
                return {"answer": f"Could not process the question due to an error: {str(e)}", "score": 0.0, "context": combined_context}
        else:
            if not context.strip():
                return {"answer": "No context provided to answer the question.", "score": 0.0, "context": ""}
            
            try:
                result = self.qa_pipeline(
                    question=question,
                    context=context,
                    top_k=topK,
                    max_answer_len=512,
                    handle_impossible_answer=True
                )
                
                if isinstance(result, list):
                    for r in result:
                        r["context"] = context
                    return result
                else:
                    result["context"] = context
                    return result
            except Exception as e:
                # Fallback for errors
                return {"answer": f"Could not process the question due to an error: {str(e)}", "score": 0.0, "context": context}
