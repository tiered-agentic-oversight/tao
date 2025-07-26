
import os
import json
import argparse
from typing import List, Dict, Any, Optional, Union
import time
import random
from collections import Counter

from utils import (
    LLMClient, TieredAgenticOversight, get_dataset,
    RiskLevel, RequiredExpertise, RouterOutput, AgentResponse,
    token_tracker
)

def parse_args():
    parser = argparse.ArgumentParser(description="Tiered Agentic Oversight System - Production Configuration")
    
    # === Dataset Configuration ===
    parser.add_argument("--dataset", type=str, required=True,
                      help="Dataset to use (medsafetybench, safetybench, mm-safety, red-teaming, medical-triage)")
    parser.add_argument("--dataset-name", type=str, required=True,
                      help="Dataset name (medsafetybench, safetybench, mm-safety, red-teaming, medical-triage)")
    parser.add_argument("--dataset-path", type=str, required=True,
                      help="Path to dataset file or directory")
    
    # === Model Configuration ===
    parser.add_argument("--router-model", type=str, required=True, 
                      help="Model to use for the router")
    parser.add_argument("--agent-model", type=str, required=True, 
                      help="Default model to use for agents")
    parser.add_argument("--tier-1-model", type=str, required=True,
                      help="Model to use for Tier 1 agents")
    parser.add_argument("--tier-2-model", type=str, required=True,
                      help="Model to use for Tier 2 agents")
    parser.add_argument("--tier-3-model", type=str, required=True,
                      help="Model to use for Tier 3 agents")
    parser.add_argument("--final-decision-model", type=str, required=True,
                      help="Model to use for final decision making")
    
    # === API Configuration ===
    parser.add_argument("--gemini-api-key", type=str, 
                      help="Gemini API key (optional, can use env var)")
    parser.add_argument("--openai-api-key", type=str, 
                      help="OpenAI API key (optional, can use env var)")
    
    # === Experimental Settings ===
    parser.add_argument("--seed", type=int, default=0, 
                      help="Random seed for reproducibility")
    parser.add_argument("--temperature", type=float, default=0.0,
                      help="LLM temperature setting")
    parser.add_argument("--max-retries", type=int, default=3,
                      help="Maximum retries for failed LLM calls")
    parser.add_argument("--initial-delay", type=int, default=1,
                      help="Initial delay for retries (seconds)")
    parser.add_argument("--timeout", type=int, default=300,
                      help="Timeout for individual cases (seconds)")
    
    # === Conversation Configuration ===
    parser.add_argument("--enable-inter-tier", action="store_true", default=False,
                      help="Enable multi-turn conversations BETWEEN different tiers")
    parser.add_argument("--enable-intra-tier", action="store_true", default=False,
                      help="Enable communication WITHIN the same tier")
    parser.add_argument("--max-turns", type=int, default=3, 
                      help="Maximum number of conversation turns per interaction")
    
    # === Decision Thresholds ===
    parser.add_argument("--escalation-threshold", type=float, default=0.5,
                      help="Threshold for automatic escalation (0.0-1.0)")
    parser.add_argument("--confidence-threshold", type=float, default=0.6,
                      help="Minimum confidence for final decisions")
    
    # === Resource Limits ===
    parser.add_argument("--max-cost-per-case", type=float, default=1.0,
                      help="Maximum cost per case (USD)")
    parser.add_argument("--max-tokens-per-case", type=int, default=100000,
                      help="Maximum tokens per case")
    
    # === Behavior Configuration ===
    parser.add_argument("--require-tier-1", action="store_true", default=True,
                      help="Always require at least Tier 1 assessment")
    parser.add_argument("--no-tier-skipping", action="store_true", default=True,
                      help="Prevent skipping tiers in escalation")
    parser.add_argument("--require-unanimous", action="store_true", default=False,
                      help="Require unanimous agreement in intra-tier discussions")
    
    # === Output Configuration ===
    parser.add_argument("--output-dir", type=str, default="outputs",
                      help="Output directory for results")
    parser.add_argument("--save-intermediate", action="store_true", default=True,
                      help="Save results after each case")
    parser.add_argument("--include-conversations", action="store_true", default=True,
                      help="Include full conversation logs")
    parser.add_argument("--enable-cost-warnings", action="store_true", default=True,
                      help="Enable real-time cost warnings")
    
    # === Verbosity and Debugging ===
    parser.add_argument("--verbose", action="store_true", 
                      help="Print verbose output")
    parser.add_argument("--debug-escalation", action="store_true", default=False,
                      help="Enable detailed escalation debugging")
    parser.add_argument("--force-multi-tier", action="store_true", default=False,
                      help="Force router to assign multiple tiers for testing")
    parser.add_argument("--test-cases", type=int, 
                      help="Limit number of test cases for debugging")
    
    return parser.parse_args()

def validate_configuration(args):
    """Validate the configuration parameters"""
    print(f"\n🔍 CONFIGURATION VALIDATION")
    print(f"{'─'*50}")
    
    if not 0.0 <= args.escalation_threshold <= 1.0:
        raise ValueError(f"Escalation threshold must be between 0.0 and 1.0, got {args.escalation_threshold}")
    
    if not 0.0 <= args.confidence_threshold <= 1.0:
        raise ValueError(f"Confidence threshold must be between 0.0 and 1.0, got {args.confidence_threshold}")
    
    if not 0.0 <= args.temperature <= 2.0:
        raise ValueError(f"Temperature must be between 0.0 and 2.0, got {args.temperature}")
    
    if args.max_cost_per_case <= 0:
        raise ValueError(f"Max cost per case must be positive, got {args.max_cost_per_case}")
    
    if args.max_tokens_per_case <= 0:
        raise ValueError(f"Max tokens per case must be positive, got {args.max_tokens_per_case}")
    
    if args.dataset_name == "medsafetybench":
        if not os.path.isdir(args.dataset_path):
            raise ValueError(f"MedSafetyBench requires directory path, got: {args.dataset_path}")
    else:
        if not os.path.isfile(args.dataset_path):
            raise ValueError(f"Dataset file not found: {args.dataset_path}")
    
    print(f"✅ Configuration validated successfully")

def process_dataset(tao, dataset_name, dataset, args, verbose=False):
    """Process any dataset using the tiered agentic oversight framework"""
    results = []
    total_cases = len(dataset) if isinstance(dataset, list) else sum(len(df) if hasattr(df, '__len__') else 1 for df in dataset.values())
    
    print(f"\n📊 DATASET PROCESSING OVERVIEW")
    print(f"{'─'*60}")
    print(f"📁 Dataset: {dataset_name}")
    print(f"📈 Total cases: {total_cases}")
    print(f"🎯 Router model: {args.router_model}")
    print(f"👨‍⚕️ Agent models: T1={args.tier_1_model}, T2={args.tier_2_model}, T3={args.tier_3_model}")
    print(f"⚖️ Final decision model: {args.final_decision_model}")
    print(f"🔄 Processing settings:")
    print(f"   • Intra-tier: {'✅ Enabled' if args.enable_intra_tier else '❌ Disabled'}")
    print(f"   • Inter-tier: {'✅ Enabled' if args.enable_inter_tier else '❌ Disabled'}")
    print(f"   • Max turns: {args.max_turns}")
    print(f"   • Temperature: {args.temperature}")
    print(f"   • Escalation threshold: {args.escalation_threshold}")
    print(f"💰 Cost limits: ${args.max_cost_per_case}/case, {args.max_tokens_per_case} tokens/case")

    case_count = 0
    
    def check_cost_limits(case_num):
        """Check if we're approaching cost limits"""
        current_cost = token_tracker.get_cost()
        cost_per_case = current_cost / max(case_num, 1)
        
        if args.enable_cost_warnings:
            if cost_per_case > args.max_cost_per_case:
                print(f"⚠️ WARNING: Cost per case (${cost_per_case:.4f}) exceeds limit (${args.max_cost_per_case})")
            
            if token_tracker.total_input_tokens + token_tracker.total_output_tokens > args.max_tokens_per_case * case_num:
                avg_tokens = (token_tracker.total_input_tokens + token_tracker.total_output_tokens) / max(case_num, 1)
                print(f"⚠️ WARNING: Tokens per case ({avg_tokens:.0f}) exceeds limit ({args.max_tokens_per_case})")

    if dataset_name == "medical-triage":
        for i, sample in enumerate(dataset):
            case_count += 1
            probe_info = sample[0]
            fairness_scores = sample[1]

            scenario_id = probe_info["scenario_id"]
            probe_id = probe_info["probe_id"]
            scenario = probe_info['scenario']
            state = probe_info["state"]
            probe_question = probe_info["probe"]
            choices = probe_info["choices"]

            print(f"\n🔍 [{i+1}/{len(dataset)}] Processing Medical-Triage Case")
            print(f"   📋 ID: {scenario_id}")
            print(f"   💰 Current cost: ${token_tracker.get_cost():.4f}")

            case_text = f"""
Scenario: {scenario}
State: {state}
Question: {probe_question}
Options:
"""
            for idx, choice in enumerate(choices):
                case_text += f"({idx}) {choice}\n"

            case_metadata = {
                "scenario_id": scenario_id,
                "probe_id": probe_id,
                "scenario": scenario,
                "state": state,
                "probe": probe_question,
                "choices": choices,
                "fairness_scores": fairness_scores
            }

            try:
                result = tao.process_case(case_text, dataset_name, case_metadata)
                check_cost_limits(case_count)
                
                if verbose:
                    tao_output = result.get('tiered_agentic_oversight_output', {})
                    router_output = tao_output.get('router_output', {})
                    final_decision = tao_output.get('final_decision', {})
                    processing_metadata = tao_output.get('processing_metadata', {})
                    
                    print(f"📊 CASE RESULTS:")
                    print(f"   🎯 Experts recruited: {len(router_output.get('required_expertise', []))}")
                    print(f"   🏆 Final risk: {final_decision.get('final_risk_level', 'unknown').upper()}")
                    print(f"   🏗️ Highest tier reached: {processing_metadata.get('last_processed_tier', 1)}")
                    print(f"   💭 Assessment: {final_decision.get('final_assessment', '')[:100]}...")
                    
                    tier_results = tao_output.get('tier_consensus_results', [])
                    for tier_result in tier_results:
                        print(f"   🔸 Tier {tier_result.get('tier')}: Risk={tier_result.get('consensus_risk_level')}, Escalate={tier_result.get('consensus_escalate')}")
                
                results.append(result)
                
                if args.save_intermediate:
                    intermediate_file = os.path.join(args.output_dir, f"intermediate_{dataset_name}_case_{i+1}.json")
                    with open(intermediate_file, 'w') as f:
                        json.dump(result, f, indent=2)
                
                if args.test_cases and i + 1 >= args.test_cases:
                    print(f"🧪 DEBUG: Reached test case limit ({args.test_cases}). Stopping.")
                    break
                
            except Exception as e:
                print(f"❌ Error processing case {i+1}: {e}")
                results.append({
                    "dataset_name": dataset_name,
                    "case_id": scenario_id,
                    "input_case": case_text,
                    "error": str(e),
                    "usage_statistics": token_tracker.get_summary()
                })

    elif dataset_name == "safetybench":
        for i, item in enumerate(dataset):
            case_count += 1
            question = item["question"]
            options = item["options"]
            case_id = item['id']
            category = item['category']

            print(f"\n🔍 [{i+1}/{len(dataset)}] Processing SafetyBench Case")
            print(f"   📋 ID: {case_id}")
            print(f"   📂 Category: {category}")
            print(f"   💰 Current cost: ${token_tracker.get_cost():.4f}")

            options_str = ""
            for idx, option in enumerate(options):
                options_str += f"({chr(ord('A') + idx)}) {option}\n"

            case_text = f"""
Category: {category}

Question: {question}

Options:
{options_str}
"""
            case_metadata = {
                "question": question,
                "options": options,
                "case_id": case_id,
                "category": category
            }

            try:
                result = tao.process_case(case_text, dataset_name, case_metadata)
                check_cost_limits(case_count)
                
                if verbose:
                    tao_output = result.get('tiered_agentic_oversight_output', {})
                    router_output = tao_output.get('router_output', {})
                    final_decision = tao_output.get('final_decision', {})
                    
                    print(f"📊 CASE RESULTS:")
                    print(f"   🎯 Experts recruited: {len(router_output.get('required_expertise', []))}")
                    print(f"   🏆 Final risk: {final_decision.get('final_risk_level', 'unknown').upper()}")
                    print(f"   💭 Assessment: {final_decision.get('final_assessment', '')[:100]}...")
                    
                    tier_results = tao_output.get('tier_consensus_results', [])
                    for tier_result in tier_results:
                        print(f"   🔸 Tier {tier_result.get('tier')}: Risk={tier_result.get('consensus_risk_level')}, Escalate={tier_result.get('consensus_escalate')}")
                            
                results.append(result)
                
                if args.save_intermediate:
                    intermediate_file = os.path.join(args.output_dir, f"intermediate_{dataset_name}_case_{i+1}.json")
                    with open(intermediate_file, 'w') as f:
                        json.dump(result, f, indent=2)
                
            except Exception as e:
                print(f"❌ Error processing case {i+1}: {e}")
                results.append({
                    "dataset_name": dataset_name,
                    "question": question,
                    "case_id": case_id,
                    "category": category,
                    "input_case": case_text,
                    "error": str(e),
                    "usage_statistics": token_tracker.get_summary()
                })

            if verbose and i == 0:
                break

    elif dataset_name == "mm-safety":
        items = list(dataset.items()) if isinstance(dataset, dict) else dataset
        for i, (key, question) in enumerate(items):
            case_count += 1
            print(f"\n🔍 [{i+1}/{len(items)}] Processing MM-Safety Case")
            print(f"   📋 ID: {key}")
            print(f"   💰 Current cost: ${token_tracker.get_cost():.4f}")
            
            case_text = question["Question"]
            case_metadata = {
                "question_id": key,
                "Question": question["Question"]
            }
            
            try:
                result = tao.process_case(case_text, dataset_name, case_metadata)
                check_cost_limits(case_count)
                
                if verbose:
                    tao_output = result.get('tiered_agentic_oversight_output', {})
                    router_output = tao_output.get('router_output', {})
                    final_decision = tao_output.get('final_decision', {})
                    
                    print(f"📊 CASE RESULTS:")
                    print(f"   🎯 Experts recruited: {len(router_output.get('required_expertise', []))}")
                    print(f"   🏆 Final risk: {final_decision.get('final_risk_level', 'unknown').upper()}")
                    print(f"   💭 Assessment: {final_decision.get('final_assessment', '')[:100]}...")
                    
                    tier_results = tao_output.get('tier_consensus_results', [])
                    for tier_result in tier_results:
                        print(f"   🔸 Tier {tier_result.get('tier')}: Risk={tier_result.get('consensus_risk_level')}, Escalate={tier_result.get('consensus_escalate')}")
                            
                results.append(result)
                
                if args.save_intermediate:
                    intermediate_file = os.path.join(args.output_dir, f"intermediate_{dataset_name}_case_{i+1}.json")
                    with open(intermediate_file, 'w') as f:
                        json.dump(result, f, indent=2)
                
            except Exception as e:
                print(f"❌ Error processing case {i+1}: {e}")
                results.append({
                    "dataset_name": dataset_name,
                    "case_id": key,
                    "input_case": case_text,
                    "error": str(e),
                    "usage_statistics": token_tracker.get_summary()
                })

    elif dataset_name == "medsafetybench":
        category_cases = []
        for category, df in dataset.items():
            for _, row in df.iterrows():
                category_cases.append((category, row))
                break 
            break  

        for i, (category, row) in enumerate(category_cases):
            case_count += 1
            harmful_request = row["harmful_medical_request"]
            print(f"\n🔍 [{i+1}/{len(category_cases)}] Processing MedSafetyBench Case")
            print(f"   📂 Category: {category}")
            print(f"   💰 Current cost: ${token_tracker.get_cost():.4f}")
            
            case_text = harmful_request
            case_metadata = {
                "harmful_request": harmful_request,
                "category": category
            }
            
            try:
                result = tao.process_case(case_text, dataset_name, case_metadata)
                check_cost_limits(case_count)
                
                if verbose:
                    tao_output = result.get('tiered_agentic_oversight_output', {})
                    router_output = tao_output.get('router_output', {})
                    final_decision = tao_output.get('final_decision', {})
                    
                    print(f"📊 CASE RESULTS:")
                    print(f"   🎯 Experts recruited: {len(router_output.get('required_expertise', []))}")
                    print(f"   🏆 Final risk: {final_decision.get('final_risk_level', 'unknown').upper()}")
                    print(f"   💭 Assessment: {final_decision.get('final_assessment', '')[:100]}...")
                    
                    tier_results = tao_output.get('tier_consensus_results', [])
                    for tier_result in tier_results:
                        print(f"   🔸 Tier {tier_result.get('tier')}: Risk={tier_result.get('consensus_risk_level')}, Escalate={tier_result.get('consensus_escalate')}")
                            
                results.append(result)
                
                if args.save_intermediate:
                    intermediate_file = os.path.join(args.output_dir, f"intermediate_{dataset_name}_case_{i+1}.json")
                    with open(intermediate_file, 'w') as f:
                        json.dump(result, f, indent=2)
                
            except Exception as e:
                print(f"❌ Error processing case {i+1}: {e}")
                results.append({
                    "dataset_name": dataset_name,
                    "case_id": harmful_request,
                    "input_case": case_text,
                    "error": str(e),
                    "usage_statistics": token_tracker.get_summary()
                })

    elif dataset_name == "red-teaming":
        for i, sample in enumerate(dataset):
            case_count += 1
            scenario_id = i
            probe_id = i
            prompt_clean = sample.get('prompt_clean', "N/A")
            prompt_text = sample.get('Prompt text', "N/A")
            additional_comments = sample.get('Additional Comments', "N/A")
            
            print(f"\n🔍 [{i+1}/{len(dataset)}] Processing Red-Teaming Case")
            print(f"   📋 ID: {scenario_id}")
            print(f"   💰 Current cost: ${token_tracker.get_cost():.4f}")
            
            case_text = f"""
Question: {prompt_clean}
Context: {prompt_text}
Additional Information: {additional_comments}
"""
            case_metadata = {
                "scenario_id": scenario_id,
                "probe_id": probe_id,
                "prompt_clean": prompt_clean,
                "prompt_text": prompt_text,
                "additional_comments": additional_comments
            }
            
            try:
                result = tao.process_case(case_text, dataset_name, case_metadata)
                check_cost_limits(case_count)
                
                if verbose:
                    tao_output = result.get('tiered_agentic_oversight_output', {})
                    router_output = tao_output.get('router_output', {})
                    final_decision = tao_output.get('final_decision', {})
                    
                    print(f"📊 CASE RESULTS:")
                    print(f"   🎯 Experts recruited: {len(router_output.get('required_expertise', []))}")
                    print(f"   🏆 Final risk: {final_decision.get('final_risk_level', 'unknown').upper()}")
                    print(f"   💭 Assessment: {final_decision.get('final_assessment', '')[:100]}...")
                    
                    tier_results = tao_output.get('tier_consensus_results', [])
                    for tier_result in tier_results:
                        print(f"   🔸 Tier {tier_result.get('tier')}: Risk={tier_result.get('consensus_risk_level')}, Escalate={tier_result.get('consensus_escalate')}")
                            
                results.append(result)
                
                if args.save_intermediate:
                    intermediate_file = os.path.join(args.output_dir, f"intermediate_{dataset_name}_case_{i+1}.json")
                    with open(intermediate_file, 'w') as f:
                        json.dump(result, f, indent=2)
                
            except Exception as e:
                print(f"❌ Error processing case {i+1}: {e}")
                results.append({
                    "dataset_name": dataset_name,
                    "case_id": scenario_id,
                    "input_case": case_text,
                    "error": str(e),
                    "usage_statistics": token_tracker.get_summary()
                })

    return results

def load_raw_dataset_data(dataset_name, dataset_path):
    """Load the raw dataset data from the specified path"""
    print(f"📁 Loading {dataset_name} from: {dataset_path}")
    
    try:
        if dataset_name in ["medical-triage", "safetybench", "mm-safety"]:
            with open(dataset_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(f"✅ Loaded {len(data) if isinstance(data, list) else len(data.keys())} items")
            return data
            
        elif dataset_name == "red-teaming":
            with open(dataset_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if 'sampled_data' in data:
                    print(f"✅ Loaded {len(data['sampled_data'])} red-teaming samples")
                    return data['sampled_data']
                else:
                    print("❌ Error: 'sampled_data' key not found in red-teaming dataset")
                    return data if isinstance(data, list) else []
                    
        elif dataset_name == "medsafetybench":
            import pandas as pd
            dataset = {}
            total_cases = 0
            
            for idx in range(1, 10):
                csv_file = f"{dataset_path}/datasets/test/gpt4/med_safety_demonstrations_category_{idx}.csv"
                try:
                    df = pd.read_csv(csv_file)
                    dataset[f"category_{idx}"] = df
                    total_cases += len(df)
                    print(f"   ✅ Category {idx}: {len(df)} cases")
                except Exception as e:
                    print(f"   ❌ Error loading category {idx}: {e}")

            print(f"✅ Total MedSafetyBench cases loaded: {total_cases}")
            return dataset
            
    except Exception as e:
        print(f"❌ Error loading dataset {dataset_name}: {e}")
        return []

def print_final_summary(results, args, start_time, end_time):
    """Print a comprehensive summary of the run"""
    processing_time = end_time - start_time
    
    if isinstance(results, list):
        num_results = len(results)
        num_errors = sum(1 for result in results if 'error' in result)
        num_tier_consensus = sum(1 for result in results if isinstance(result, dict) and 'tiered_agentic_oversight_output' in result and
                                 isinstance(result['tiered_agentic_oversight_output'], dict) and
                                 'tier_consensus_results' in result['tiered_agentic_oversight_output'])
        num_valid_results = num_results - num_errors
    else:
        num_results = 0
        num_errors = 0
        num_tier_consensus = 0
        num_valid_results = 0

    token_stats = token_tracker.get_summary()
    
    print(f"\n" + "="*80)
    print(f"📊 FINAL RUN SUMMARY")
    print(f"="*80)
    print(f"🗂️  Dataset: {args.dataset} (processed as {args.dataset_name})")
    print(f"📁 Dataset path: {args.dataset_path}")
    print(f"🎲 Seed: {args.seed}")
    
    print(f"\n⚙️  Model Configuration:")
    print(f"   • Router: {args.router_model}")
    print(f"   • Tier 1: {args.tier_1_model}")
    print(f"   • Tier 2: {args.tier_2_model}")
    print(f"   • Tier 3: {args.tier_3_model}")
    print(f"   • Final Decision: {args.final_decision_model}")
    
    print(f"\n🔄 Conversation Settings:")
    print(f"   • Intra-tier: {'✅ Enabled' if args.enable_intra_tier else '❌ Disabled'}")
    print(f"   • Inter-tier: {'✅ Enabled' if args.enable_inter_tier else '❌ Disabled'}")
    print(f"   • Max turns: {args.max_turns}")
    print(f"   • Temperature: {args.temperature}")
    print(f"   • Escalation threshold: {args.escalation_threshold}")
    print(f"   • Confidence threshold: {args.confidence_threshold}")
    
    print(f"\n📈 Processing Results:")
    print(f"   • Total cases attempted: {num_results}")
    print(f"   • Successfully processed: {num_valid_results}")
    print(f"   • Processing errors: {num_errors}")
    print(f"   • Cases with tier consensus: {num_tier_consensus}")
    
    print(f"\n⏱️  Performance Metrics:")
    print(f"   • Total processing time: {processing_time:.2f}s")
    print(f"   • Average time per case: {processing_time/max(num_results, 1):.2f}s")
    print(f"   • Total LLM calls: {token_stats['total_calls']}")
    print(f"   • Total inference time: {token_stats['total_inference_time_seconds']}s")
    print(f"   • Average inference time per call: {token_stats['avg_inference_time_per_call']}s")
    
    print(f"\n💰 Cost Analysis:")
    print(f"   • Input tokens: {token_stats['total_input_tokens']:,}")
    print(f"   • Output tokens: {token_stats['total_output_tokens']:,}")
    print(f"   • Total tokens: {token_stats['total_tokens']:,}")
    print(f"   • Estimated cost: ${token_stats['estimated_cost_usd']:.4f}")
    
    if num_valid_results > 0:
        print(f"   • Cost per case: ${token_stats['estimated_cost_usd']/num_valid_results:.4f}")
        print(f"   • Tokens per case: {token_stats['total_tokens']//num_valid_results:,}")
    
    if num_valid_results > 0:
        cost_per_case = token_stats['estimated_cost_usd'] / num_valid_results
        tokens_per_case = token_stats['total_tokens'] // num_valid_results
        
        print(f"\n⚠️  Resource Usage vs Limits:")
        cost_status = "✅" if cost_per_case <= args.max_cost_per_case else "❌"
        token_status = "✅" if tokens_per_case <= args.max_tokens_per_case else "❌"
        print(f"   • Cost per case: {cost_status} ${cost_per_case:.4f} (limit: ${args.max_cost_per_case})")
        print(f"   • Tokens per case: {token_status} {tokens_per_case:,} (limit: {args.max_tokens_per_case:,})")
    
    if num_valid_results > 0:
        risk_distribution = {"LOW": 0, "MEDIUM": 0, "HIGH": 0, "CRITICAL": 0}
        tier_distribution = {1: 0, 2: 0, 3: 0}
        
        for result in results:
            if 'error' not in result and 'tiered_agentic_oversight_output' in result:
                tao_output = result['tiered_agentic_oversight_output']
                final_decision = tao_output.get('final_decision', {})
                final_risk = final_decision.get('final_risk_level', 'medium').upper()
                if final_risk in risk_distribution:
                    risk_distribution[final_risk] += 1
                
                processing_metadata = tao_output.get('processing_metadata', {})
                max_tier = processing_metadata.get('last_processed_tier', 1)
                if max_tier in tier_distribution:
                    tier_distribution[max_tier] += 1
        
        print(f"\n📊 Risk Distribution:")
        for risk, count in risk_distribution.items():
            percentage = (count/num_valid_results)*100 if num_valid_results > 0 else 0
            print(f"   • {risk}: {count} cases ({percentage:.1f}%)")
        
        print(f"\n🏆 Tier Usage:")
        for tier, count in tier_distribution.items():
            percentage = (count/num_valid_results)*100 if num_valid_results > 0 else 0
            print(f"   • Highest tier {tier}: {count} cases ({percentage:.1f}%)")
    
    print(f"\n" + "="*80)

def run(args):
    validate_configuration(args)
    
    intra_tier_flag = "_intra" if args.enable_intra_tier else ""
    inter_tier_flag = "_inter" if args.enable_inter_tier else ""
    max_turns_info = f"_turns{args.max_turns}"
    
    output_file = f"{args.dataset}_tao_seed{args.seed}{intra_tier_flag}{inter_tier_flag}{max_turns_info}.json"
    output_path = os.path.join(args.output_dir, output_file)

    if os.path.exists(output_path):
        print(f"⚠️ Output file already exists: {output_file}")
        print("Skipping to avoid overwriting existing results.")
        return
    
    random.seed(args.seed)
    start_time = time.time()
    llm_client = LLMClient(
        api_key=args.gemini_api_key, 
        openai_key=args.openai_api_key, 
        seed=args.seed
    )

    print(f"\n📁 DATASET LOADING")
    print(f"{'─'*40}")
    raw_data = load_raw_dataset_data(args.dataset, args.dataset_path)
    if not raw_data:
        print(f"❌ Failed to load dataset: {args.dataset}")
        return

    tier_model_mapping = {
        1: args.tier_1_model,
        2: args.tier_2_model,
        3: args.tier_3_model
    }

    print(f"\n🏥 TAO SYSTEM INITIALIZATION")
    print(f"{'─'*40}")
    tao = TieredAgenticOversight(
        llm_client=llm_client,
        router_model=args.router_model,
        agent_model=args.agent_model,
        tier_model_mapping=tier_model_mapping,
        final_decision_model=args.final_decision_model,
        seed=args.seed,
        max_conversation_turns=args.max_turns,
        enable_inter_tier=args.enable_inter_tier,
        enable_intra_tier=args.enable_intra_tier,
        temperature=args.temperature,
        max_retries=args.max_retries,
        initial_delay=args.initial_delay,
        escalation_threshold=args.escalation_threshold,
        confidence_threshold=args.confidence_threshold,
        require_tier_1=args.require_tier_1,
        allow_tier_skipping=not args.no_tier_skipping,
        require_unanimous_consensus=args.require_unanimous,
        debug_escalation=args.debug_escalation,
        force_multi_tier=args.force_multi_tier
    )

    if args.test_cases:
        print(f"🧪 DEBUG MODE: Limiting to {args.test_cases} test cases")
        if isinstance(raw_data, list):
            raw_data = raw_data[:args.test_cases]
        elif isinstance(raw_data, dict):
            limited_data = {}
            case_count = 0
            for category, df in raw_data.items():
                if case_count >= args.test_cases:
                    break
                limited_data[category] = df.head(args.test_cases - case_count)
                case_count += len(limited_data[category])
            raw_data = limited_data

    print(f"\n🔄 DATASET PROCESSING")
    print(f"{'─'*40}")
    results = process_dataset(tao, args.dataset_name, raw_data, args, args.verbose)

    end_time = time.time()

    print(f"\n💾 SAVING RESULTS")
    print(f"{'─'*40}")
    os.makedirs(args.output_dir, exist_ok=True)
    
    run_metadata = {
        "run_configuration": {
            "dataset": args.dataset,
            "dataset_name": args.dataset_name,
            "dataset_path": args.dataset_path,
            "models": {
                "router": args.router_model,
                "agent_default": args.agent_model,
                "tier_1": args.tier_1_model,
                "tier_2": args.tier_2_model,
                "tier_3": args.tier_3_model,
                "final_decision": args.final_decision_model
            },
            "conversation_settings": {
                "enable_intra_tier": args.enable_intra_tier,
                "enable_inter_tier": args.enable_inter_tier,
                "max_turns": args.max_turns,
                "temperature": args.temperature
            },
            "thresholds": {
                "escalation_threshold": args.escalation_threshold,
                "confidence_threshold": args.confidence_threshold
            },
            "resource_limits": {
                "max_cost_per_case": args.max_cost_per_case,
                "max_tokens_per_case": args.max_tokens_per_case
            },
            "behavior_settings": {
                "require_tier_1": args.require_tier_1,
                "allow_tier_skipping": not args.no_tier_skipping,
                "require_unanimous_consensus": args.require_unanimous
            },
            "experimental_settings": {
                "seed": args.seed,
                "max_retries": args.max_retries,
                "initial_delay": args.initial_delay,
                "timeout": args.timeout
            },
            "total_processing_time_seconds": round(end_time - start_time, 2)
        },
        "usage_statistics": token_tracker.get_summary(),
        "results": results
    }
    
    with open(output_path, "w") as f:
        json.dump(run_metadata, f, indent=2)
    
    print(f"✅ Results saved to: {output_path}")
    print_final_summary(results, args, start_time, end_time)

if __name__ == "__main__":
    args = parse_args()
    
    if args.dataset != args.dataset_name:
         print(f"⚠️ Warning: --dataset '{args.dataset}' and --dataset-name '{args.dataset_name}' differ")
         print("Ensure this is intentional.")
    
    try:
        run(args)
    except KeyboardInterrupt:
        print(f"\n⚠️ Run interrupted by user")
        print(f"💰 Current estimated cost: ${token_tracker.get_cost():.4f}")
    except Exception as e:
        print(f"\n❌ Run failed with error: {e}")
        print(f"💰 Current estimated cost: ${token_tracker.get_cost():.4f}")
        raise
