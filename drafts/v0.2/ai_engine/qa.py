#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Question answering module using transformers pipeline.
"""

from typing import Dict, List, Tuple, Union, Any
import numpy as np
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline


class QuestionAnswerer:
    """
    A class for answering questions using a transformer-based QA model.
    """
    
    def __init__(self, model_name: str = "deepset/roberta-base-squad2") -> None:
        """
        Initialize the QuestionAnswerer.
        
        Args:
            model_name: Name of the QA model to use
        """
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForQuestionAnswering.from_pretrained(model_name)
            self.qa_pipeline = pipeline(
                "question-answering",
                model=self.model,
                tokenizer=self.tokenizer,
                handle_impossible_answer=True
            )
        except Exception as e:
            raise Exception(f"Failed to load QA model '{model_name}': {e}")
    
    def answerQuestion(self, question: str, contexts: List[str], top_k: int = 1) -> List[Dict[str, Any]]:
        """
        Answer a question based on given contexts.
        
        Args:
            question: The question to answer
            contexts: List of context strings to use for answering
            top_k: Number of answers to return
            
        Returns:
            List of dicts containing answer information:
              - 'answer': The answer text
              - 'score': Confidence score
              - 'context': The context snippet used
              - 'start': Start position in context
              - 'end': End position in context
        """
        if not contexts:
            return [{
                'answer': 'No context provided to answer the question.',
                'score': 0.0,
                'context': '',
                'start': 0,
                'end': 0
            }]
        
        try:
            # Process each context individually
            all_answers = []
            
            for context in contexts:
                # Skip empty contexts
                if not context.strip():
                    continue
                
                result = self.qa_pipeline(
                    question=question,
                    context=context,
                    handle_impossible_answer=True,
                    max_answer_len=100
                )
                
                # Convert to standardized format
                answer = {
                    'answer': result['answer'],
                    'score': result['score'],
                    'context': context,
                    'start': result['start'],
                    'end': result['end']
                }
                
                all_answers.append(answer)
            
            # Sort by score and take top_k
            all_answers = sorted(all_answers, key=lambda x: x['score'], reverse=True)[:top_k]
            
            # If no good answers are found
            if not all_answers or all_answers[0]['score'] < 0.01:
                return [{
                    'answer': 'I could not find an answer to this question in the provided context.',
                    'score': 0.0,
                    'context': contexts[0] if contexts else '',
                    'start': 0,
                    'end': 0
                }]
            
            return all_answers
            
        except Exception as e:
            return [{
                'answer': f'Error answering question: {e}',
                'score': 0.0,
                'context': contexts[0] if contexts else '',
                'start': 0,
                'end': 0
            }]
    
    def answerWithRetrievedContext(self, 
                                   question: str, 
                                   retrieved_documents: List[Dict[str, Any]], 
                                   use_separate_contexts: bool = True,
                                   top_k_answers: int = 1) -> List[Dict[str, Any]]:
        """
        Answer a question using retrieved document chunks.
        
        Args:
            question: The question to answer
            retrieved_documents: List of retrieved document chunks with their metadata
            use_separate_contexts: Whether to process each context separately (True) 
                                  or combine them (False)
            top_k_answers: Number of answers to return
            
        Returns:
            List of dicts containing answer information with additional metadata:
              - 'answer': The answer text
              - 'score': Confidence score
              - 'context': The context snippet used
              - 'start': Start position in context
              - 'end': End position in context
              - 'doc_id': Document ID
              - 'page': Page number
              - 'chunk_id': Chunk ID
              - 'retrieval_score': Retrieval similarity score
        """
        if not retrieved_documents:
            return [{
                'answer': 'No documents were retrieved to answer the question.',
                'score': 0.0,
                'context': '',
                'start': 0,
                'end': 0
            }]
        
        # Extract contexts from retrieved documents
        if use_separate_contexts:
            # Process each context separately
            all_answers = []
            
            for doc in retrieved_documents:
                context = doc['text']
                
                # Skip empty contexts
                if not context.strip():
                    continue
                
                result = self.qa_pipeline(
                    question=question,
                    context=context,
                    handle_impossible_answer=True,
                    max_answer_len=100
                )
                
                # Convert to standardized format with metadata
                answer = {
                    'answer': result['answer'],
                    'score': result['score'],
                    'context': context,
                    'start': result['start'],
                    'end': result['end'],
                    'doc_id': doc['doc_id'],
                    'page': doc['page'],
                    'chunk_id': doc['chunk_id'],
                    'retrieval_score': doc['score'],
                    'doc_info': doc['doc_info']
                }
                
                all_answers.append(answer)
            
            # Sort by score and take top_k
            all_answers = sorted(all_answers, key=lambda x: x['score'], reverse=True)[:top_k_answers]
            
            # If no good answers are found
            if not all_answers or all_answers[0]['score'] < 0.01:
                return [{
                    'answer': 'I could not find an answer to this question in the retrieved documents.',
                    'score': 0.0,
                    'context': retrieved_documents[0]['text'],
                    'start': 0,
                    'end': 0,
                    'doc_id': retrieved_documents[0]['doc_id'],
                    'page': retrieved_documents[0]['page'],
                    'chunk_id': retrieved_documents[0]['chunk_id'],
                    'retrieval_score': retrieved_documents[0]['score'],
                    'doc_info': retrieved_documents[0]['doc_info']
                }]
            
            return all_answers
        else:
            # Combine contexts into one
            combined_context = " ".join(doc['text'] for doc in retrieved_documents)
            
            result = self.qa_pipeline(
                question=question,
                context=combined_context,
                handle_impossible_answer=True,
                max_answer_len=100
            )
            
            # Find which document the answer came from
            answer_text = result['answer']
            answer_start = result['start']
            answer_end = result['end']
            
            current_pos = 0
            source_doc = retrieved_documents[0]  # Default to first doc
            
            for doc in retrieved_documents:
                context_length = len(doc['text']) + 1  # +1 for the space
                if current_pos <= answer_start < current_pos + context_length:
                    source_doc = doc
                    break
                current_pos += context_length
            
            # Convert to standardized format with metadata
            answer = {
                'answer': answer_text,
                'score': result['score'],
                'context': combined_context,
                'start': answer_start,
                'end': answer_end,
                'doc_id': source_doc['doc_id'],
                'page': source_doc['page'],
                'chunk_id': source_doc['chunk_id'],
                'retrieval_score': source_doc['score'],
                'doc_info': source_doc['doc_info']
            }
            
            return [answer]


if __name__ == "__main__":
    # Sample usage
    qa = QuestionAnswerer()
    
    test_question = "What is a transformer model?"
    test_contexts = [
        "The transformer model is a deep learning model introduced in 2017 that utilizes self-attention mechanisms.",
        "Transformers have revolutionized NLP tasks by enabling parallel processing of input sequences.",
        "Unlike recurrent neural networks, transformer models don't process data sequentially."
    ]
    
    answers = qa.answerQuestion(test_question, test_contexts)
    
    print(f"Question: {test_question}")
    print("\nAnswers:")
    
    for i, answer in enumerate(answers):
        print(f"\nAnswer {i+1}:")
        print(f"  Text: {answer['answer']}")
        print(f"  Score: {answer['score']:.4f}")
        print(f"  Context: {answer['context']}")
