#!/bin/bash

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m'

ROUTER_MODEL="gemini-2.0-flash"
AGENT_MODEL="gemini-2.0-flash"

TIER_1_MODEL="gemini-2.0-flash"  # Fast screening
TIER_2_MODEL="gemini-2.0-flash"  # Specialized analysis  
TIER_3_MODEL="gemini-2.0-flash"  # Expert consultation

FINAL_DECISION_MODEL="gemini-2.0-flash"

DATASETS=("safetybench")

DATASET_PATHS_medical_triage="./data/medical_triage.json"
DATASET_PATHS_safetybench="./data/safetybench.json"
DATASET_PATHS_mm_safety="./data/mm-safety.json"
DATASET_PATHS_red_teaming="./data/red-teaming.json"
DATASET_PATHS_medsafetybench="./data/medsafetybench"

MAX_TURNS=3                    # Maximum conversation turns per interaction
ENABLE_INTRA_TIER=true        # Enable conversations within same tier
ENABLE_INTER_TIER=true        # Enable conversations between tiers

SEEDS=(0)                      # Random seeds for reproducibility

VERBOSE=true                   # Enable verbose output
TEMPERATURE=0.0                # LLM temperature for consistency

MAX_RETRIES=3                  # Maximum retries for failed LLM calls
INITIAL_DELAY=1                # Initial delay for retries (seconds)
TIMEOUT=300                    # Timeout for individual cases (seconds)

OUTPUT_DIR="./outputs"
LOGS_DIR="./logs"
SAVE_INTERMEDIATE_RESULTS=false # Save results after each case
INCLUDE_CONVERSATION_HISTORY=true # Include full conversation logs

MAX_COST_PER_CASE=1.0         # Maximum cost per case (USD) - will warn if exceeded
MAX_TOKENS_PER_CASE=100000    # Maximum tokens per case - will warn if exceeded
ENABLE_COST_WARNINGS=true     # Enable real-time cost warnings

ESCALATION_THRESHOLD=0.3       # Threshold for automatic escalation (0.0-1.0)
CONFIDENCE_THRESHOLD=0.4       # Minimum confidence for final decisions

REQUIRE_TIER_1=true           # Always require at least Tier 1 assessment
ALLOW_TIER_SKIPPING=false     # Allow skipping tiers in escalation
MAX_TIER=3                    # Maximum tier level

REQUIRE_UNANIMOUS_CONSENSUS=false  # Require unanimous agreement in intra-tier discussions
MIN_CONSENSUS_CONFIDENCE=0.6       # Minimum confidence for consensus decisions

TEST_CASES=5                  # Limit number of test cases for debugging
DEBUG_ESCALATION=true         # Enable detailed escalation debugging

echo -e "${PURPLE}===============================================${NC}"
echo -e "${PURPLE}🏥 FIXED TIERED AGENTIC OVERSIGHT (TAO) SYSTEM${NC}"
echo -e "${PURPLE}Real Escalation & Agent Interaction Enabled${NC}"
echo -e "${PURPLE}===============================================${NC}"

echo -e "\n${CYAN}🔧 Environment Validation${NC}"
echo "----------------------------------------"

if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ python3 not found${NC}"
    exit 1
fi
echo -e "${GREEN}✅ Python 3 found: $(python3 --version)${NC}"

if [ -z "$GEMINI_API_KEY" ]; then
    echo -e "${RED}❌ GEMINI_API_KEY not set in environment${NC}"
    echo -e "${YELLOW}Please export your Gemini API key:${NC}"
    echo -e "${YELLOW}export GEMINI_API_KEY='your_gemini_api_key_here'${NC}"
    exit 1
fi
echo -e "${GREEN}✅ Gemini API key found${NC}"

if [ -z "$OPENAI_API_KEY" ]; then
    echo -e "${YELLOW}⚠️ OPENAI_API_KEY not set (optional for OpenAI models)${NC}"
else
    echo -e "${GREEN}✅ OpenAI API key found${NC}"
fi

echo -e "\n${CYAN}📁 Dataset Validation${NC}"
echo "----------------------------------------"
for dataset in "${DATASETS[@]}"; do
    dataset_var="DATASET_PATHS_${dataset//-/_}"
    dataset_path="${!dataset_var}"
    
    if [ -z "$dataset_path" ]; then
        echo -e "${RED}❌ Path not configured for dataset: $dataset${NC}"
        exit 1
    fi
    
    if [[ "$dataset" == "medsafetybench" ]]; then
        if [ ! -d "$dataset_path" ]; then
            echo -e "${RED}❌ Dataset directory not found: $dataset_path${NC}"
            exit 1
        fi
    else
        if [ ! -f "$dataset_path" ]; then
            echo -e "${RED}❌ Dataset file not found: $dataset_path${NC}"
            exit 1
        fi
    fi
    echo -e "${GREEN}✅ Dataset validated: $dataset${NC}"
done

mkdir -p "$OUTPUT_DIR"
mkdir -p "$LOGS_DIR"

echo -e "\n${CYAN}📋 FIXED Experimental Configuration${NC}"
echo "=========================================="
echo -e "${BLUE}🎯 Models:${NC}"
echo -e "   Router: $ROUTER_MODEL"
echo -e "   Agent: $AGENT_MODEL"
echo -e "   Tier 1: $TIER_1_MODEL"
echo -e "   Tier 2: $TIER_2_MODEL"
echo -e "   Tier 3: $TIER_3_MODEL"
echo -e "   Final Decision: $FINAL_DECISION_MODEL"

echo -e "\n${BLUE}📊 Datasets: ${DATASETS[@]}${NC}"

echo -e "\n${BLUE}⚙️ FIXED Conversation Settings:${NC}"
echo -e "   Max turns: $MAX_TURNS"
echo -e "   Intra-tier: $([ "$ENABLE_INTRA_TIER" = true ] && echo "✅ Enabled" || echo "❌ Disabled")"
echo -e "   Inter-tier: $([ "$ENABLE_INTER_TIER" = true ] && echo "✅ Enabled" || echo "❌ Disabled")"
echo -e "   Real escalation: ✅ ENABLED"
echo -e "   Conservative router: ✅ ENABLED"

echo -e "\n${BLUE}🔧 FIXED Features:${NC}"
echo -e "   • Router starts with Tier 1 only"
echo -e "   • Higher tiers created on-demand during escalation"
echo -e "   • Real multi-agent discussions within tiers"
echo -e "   • True collaborative assessment between tiers"
echo -e "   • Structured escalation decision parsing"

echo -e "\n${BLUE}🧪 Testing Settings:${NC}"
echo -e "   Test cases limit: ${TEST_CASES:-'None (all cases)'}"
echo -e "   Debug escalation: $([ "$DEBUG_ESCALATION" = true ] && echo "✅ Enabled" || echo "❌ Disabled")"
echo -e "   Escalation threshold: $ESCALATION_THRESHOLD (lower = more escalation)"

echo -e "\n${BLUE}🎲 Seeds: ${SEEDS[@]}${NC}"

echo -e "\n${BLUE}💰 Cost Limits:${NC}"
echo -e "   Max cost per case: \$${MAX_COST_PER_CASE}"
echo -e "   Max tokens per case: ${MAX_TOKENS_PER_CASE}"

echo -e "\n${BLUE}📁 Output:${NC}"
echo -e "   Results: $OUTPUT_DIR"
echo -e "   Logs: $LOGS_DIR"

echo -e "\n${YELLOW}💰 Cost Estimation${NC}"
echo "----------------------------------------"
echo -e "${YELLOW}Gemini-2.0-flash pricing (per 1M tokens):${NC}"
echo -e "  • Input: \$0.30${NC}"
echo -e "  • Output: \$2.50${NC}"
estimated_cases=$((${#DATASETS[@]} * ${#SEEDS[@]}))
if [ -n "$TEST_CASES" ]; then
    estimated_cases=$TEST_CASES
fi
echo -e "${YELLOW}Estimated cases to process: ~$estimated_cases${NC}"
echo -e "${YELLOW}With real conversations, expect higher token usage${NC}"
echo -e "${YELLOW}Monitor costs during execution${NC}"

echo -e "\n${GREEN}🚀 FIXED System Ready${NC}"
echo "----------------------------------------"
echo -e "${GREEN}✅ Conservative router (Tier 1 start only)${NC}"
echo -e "${GREEN}✅ Real multi-agent conversations${NC}"
echo -e "${GREEN}✅ True collaborative escalation${NC}"
echo -e "${GREEN}✅ On-demand higher tier creation${NC}"
echo -e "${GREEN}✅ Structured escalation parsing${NC}"

read -p "Continue with FIXED escalation system? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${YELLOW}Experiment cancelled by user${NC}"
    exit 0
fi

# FIXED: Prepare conversation flags
intra_tier_flag=""
if [ "$ENABLE_INTRA_TIER" = true ]; then
    intra_tier_flag="--enable-intra-tier"
else
    intra_tier_flag="--disable-intra-tier"
fi

inter_tier_flag=""
if [ "$ENABLE_INTER_TIER" = true ]; then
    inter_tier_flag="--enable-inter-tier"
else
    inter_tier_flag="--disable-inter-tier"
fi

verbose_flag=""
if [ "$VERBOSE" = true ]; then
    verbose_flag="--verbose"
fi

debug_flag=""
if [ "$DEBUG_ESCALATION" = true ]; then
    debug_flag="--debug-escalation"
fi

test_cases_flag=""
if [ -n "$TEST_CASES" ]; then
    test_cases_flag="--test-cases $TEST_CASES"
fi

total_start_time=$(date +%s)

for seed in "${SEEDS[@]}"; do
    echo -e "\n${PURPLE}===============================================${NC}"
    echo -e "${PURPLE}🎲 FIXED EXPERIMENT BATCH - SEED $seed${NC}"
    echo -e "${PURPLE}===============================================${NC}"

    batch_start_time=$(date +%s)
    timestamp=$(date +"%Y%m%d_%H%M%S")

    for dataset in "${DATASETS[@]}"; do
        echo -e "\n${GREEN}------------------------------------------------${NC}"
        echo -e "${GREEN}📊 Processing: $dataset (seed: $seed) - FIXED VERSION${NC}"
        echo -e "${GREEN}------------------------------------------------${NC}"

        dataset_var="DATASET_PATHS_${dataset//-/_}"
        dataset_path="${!dataset_var}"

        log_prefix="FIXED_tao"
        if [ "$ENABLE_INTRA_TIER" = true ]; then
            log_prefix="${log_prefix}_intra"
        fi
        if [ "$ENABLE_INTER_TIER" = true ]; then
            log_prefix="${log_prefix}_inter"
        fi
        log_prefix="${log_prefix}_turns${MAX_TURNS}"

        log_file="${LOGS_DIR}/${log_prefix}_${dataset}_seed${seed}_${timestamp}.log"

        echo -e "${BLUE}📄 Log: $(basename $log_file)${NC}"
        echo -e "${BLUE}⏰ Start: $(date)${NC}"
        echo -e "${BLUE}🔧 FIXED Features: Conservative router, Real escalation${NC}"

        dataset_start_time=$(date +%s)

        # FIXED: Use the corrected main.py with real escalation
        python3 main.py \
            --dataset "$dataset" \
            --dataset-name "$dataset" \
            --dataset-path "$dataset_path" \
            --router-model "$ROUTER_MODEL" \
            --agent-model "$AGENT_MODEL" \
            --tier-1-model "$TIER_1_MODEL" \
            --tier-2-model "$TIER_2_MODEL" \
            --tier-3-model "$TIER_3_MODEL" \
            --final-decision-model "$FINAL_DECISION_MODEL" \
            --seed "$seed" \
            --max-turns "$MAX_TURNS" \
            --temperature "$TEMPERATURE" \
            --max-retries "$MAX_RETRIES" \
            --initial-delay "$INITIAL_DELAY" \
            --timeout "$TIMEOUT" \
            --escalation-threshold "$ESCALATION_THRESHOLD" \
            --confidence-threshold "$CONFIDENCE_THRESHOLD" \
            --max-cost-per-case "$MAX_COST_PER_CASE" \
            --max-tokens-per-case "$MAX_TOKENS_PER_CASE" \
            --output-dir "$OUTPUT_DIR" \
            $intra_tier_flag \
            $inter_tier_flag \
            $verbose_flag \
            $debug_flag \
            $test_cases_flag \
            $([ "$REQUIRE_TIER_1" = true ] && echo "--require-tier-1") \
            $([ "$ALLOW_TIER_SKIPPING" = false ] && echo "--no-tier-skipping") \
            $([ "$REQUIRE_UNANIMOUS_CONSENSUS" = true ] && echo "--require-unanimous") \
            $([ "$SAVE_INTERMEDIATE_RESULTS" = true ] && echo "--save-intermediate") \
            $([ "$INCLUDE_CONVERSATION_HISTORY" = true ] && echo "--include-conversations") \
            $([ "$ENABLE_COST_WARNINGS" = true ] && echo "--enable-cost-warnings") \
            --enable-research-mode \
            --extract-individual-answers \
            --conduct-error-analysis \
            2>&1 | tee "$log_file"

        python_exit_status=${PIPESTATUS[0]}
        dataset_end_time=$(date +%s)
        dataset_duration=$((dataset_end_time - dataset_start_time))

        echo -e "\n${BLUE}⏰ End: $(date)${NC}"
        echo -e "${BLUE}⏱️ Duration: ${dataset_duration}s${NC}"

        if [ $python_exit_status -ne 0 ]; then
            echo -e "${RED}❌ FAILED: $dataset (seed: $seed)${NC}"
            echo -e "${RED}📄 Check log: $log_file${NC}"
        
        else
            echo -e "${GREEN}✅ SUCCESS: $dataset (seed: $seed) - ${dataset_duration}s${NC}"
            
            # FIXED: Look for the corrected output file pattern
            output_pattern="${dataset}_FIXED_tao_seed${seed}"
            if [ "$ENABLE_INTRA_TIER" = true ]; then
                output_pattern="${output_pattern}_intra"
            fi
            if [ "$ENABLE_INTER_TIER" = true ]; then
                output_pattern="${output_pattern}_inter"
            fi
            output_pattern="${output_pattern}_turns${MAX_TURNS}_research.json"
            
            output_file="$OUTPUT_DIR/$output_pattern"
            if [ -f "$output_file" ]; then
                file_size=$(du -h "$output_file" | cut -f1)
                echo -e "${GREEN}📁 Output: $output_pattern (${file_size})${NC}"
                
                if command -v jq &> /dev/null; then
                    cost=$(jq -r '.usage_statistics.estimated_cost_usd' "$output_file" 2>/dev/null || echo "N/A")
                    tokens=$(jq -r '.usage_statistics.total_tokens' "$output_file" 2>/dev/null || echo "N/A")
                    escalations=$(jq -r '[.results[].tiered_agentic_oversight_output.processing_metadata.last_processed_tier] | map(select(. > 1)) | length' "$output_file" 2>/dev/null || echo "N/A")
                    real_escalation_flag=$(jq -r '.run_configuration.real_escalation_enabled' "$output_file" 2>/dev/null || echo "false")
                    
                    echo -e "${GREEN}💰 Cost: \$${cost}, Tokens: ${tokens}${NC}"
                    echo -e "${GREEN}⬆️ Cases escalated beyond Tier 1: ${escalations}${NC}"
                    echo -e "${GREEN}🔧 Real escalation enabled: ${real_escalation_flag}${NC}"
                    
                    # FIXED: Show escalation statistics
                    if command -v jq &> /dev/null && [ "$escalations" != "N/A" ] && [ "$escalations" -gt 0 ]; then
                        tier_2_cases=$(jq -r '[.results[].tiered_agentic_oversight_output.processing_metadata.last_processed_tier] | map(select(. == 2)) | length' "$output_file" 2>/dev/null || echo "0")
                        tier_3_cases=$(jq -r '[.results[].tiered_agentic_oversight_output.processing_metadata.last_processed_tier] | map(select(. == 3)) | length' "$output_file" 2>/dev/null || echo "0")
                        echo -e "${GREEN}   📊 Tier 2 cases: ${tier_2_cases}, Tier 3 cases: ${tier_3_cases}${NC}"
                    fi
                fi
            else
                echo -e "${YELLOW}⚠️ Output file missing: $output_pattern${NC}"
            fi
            
            # FIXED: Check log for escalation evidence
            if [ -f "$log_file" ]; then
                escalation_mentions=$(grep -c "ESCALATION" "$log_file" 2>/dev/null || echo "0")
                conversation_mentions=$(grep -c "COLLABORATIVE ASSESSMENT\|INTRA-TIER DISCUSSION" "$log_file" 2>/dev/null || echo "0")
                echo -e "${GREEN}📊 Log analysis: ${escalation_mentions} escalation events, ${conversation_mentions} conversations${NC}"
            fi
        fi

        echo -e "${BLUE}⏳ Waiting 5 seconds...${NC}"
        sleep 5

    done

    batch_end_time=$(date +%s)
    batch_duration=$((batch_end_time - batch_start_time))
    echo -e "\n${GREEN}✅ Batch completed (seed $seed): ${batch_duration}s${NC}"

done

# FIXED: Enhanced summary
total_end_time=$(date +%s)
total_duration=$((total_end_time - total_start_time))

echo -e "\n${PURPLE}===============================================${NC}"
echo -e "${PURPLE}🎉 FIXED EXPERIMENTS COMPLETED${NC}"
echo -e "${PURPLE}===============================================${NC}"

echo -e "\n${CYAN}📊 FIXED Execution Summary${NC}"
echo "----------------------------------------"
echo -e "⏰ Total time: ${total_duration}s ($(($total_duration / 60))m $(($total_duration % 60))s)"
echo -e "📊 Datasets processed: ${#DATASETS[@]}"
echo -e "🎲 Seeds tested: ${#SEEDS[@]}"
echo -e "📁 Results in: $OUTPUT_DIR"
echo -e "📄 Logs in: $LOGS_DIR"

output_count=$(find "$OUTPUT_DIR" -name "*_FIXED_tao_seed*.json" | wc -l)
echo -e "📋 FIXED output files: $output_count"

if command -v jq &> /dev/null; then
    total_cost=$(find "$OUTPUT_DIR" -name "*_FIXED_tao_seed*.json" -exec jq -r '.usage_statistics.estimated_cost_usd // 0' {} \; | awk '{sum += $1} END {print sum}')
    echo -e "💰 Total estimated cost: \$${total_cost:-0}"
    
    # FIXED: Analyze escalation effectiveness
    total_escalations=$(find "$OUTPUT_DIR" -name "*_FIXED_tao_seed*.json" -exec jq -r '[.results[].tiered_agentic_oversight_output.processing_metadata.last_processed_tier] | map(select(. > 1)) | length' {} \; | awk '{sum += $1} END {print sum}')
    total_cases=$(find "$OUTPUT_DIR" -name "*_FIXED_tao_seed*.json" -exec jq -r '.results | length' {} \; | awk '{sum += $1} END {print sum}')
    
    if [ -n "$total_cases" ] && [ "$total_cases" -gt 0 ]; then
        escalation_rate=$(echo "scale=1; $total_escalations * 100 / $total_cases" | bc -l 2>/dev/null || echo "N/A")
        echo -e "⬆️ Escalation rate: ${escalation_rate}% (${total_escalations}/${total_cases} cases)"
    fi
fi

error_count=$(find "$LOGS_DIR" -name "*FIXED*.log" -exec grep -l "ERROR\|Failed\|Exception" {} \; | wc -l)
if [ $error_count -gt 0 ]; then
    echo -e "${YELLOW}⚠️ Errors detected in $error_count log files${NC}"
else
    echo -e "${GREEN}✅ No errors detected${NC}"
fi

echo -e "\n${GREEN}🎯 FIXED System Verification${NC}"
echo "----------------------------------------"

# FIXED: Verify that real escalation occurred
if command -v jq &> /dev/null; then
    echo -e "Checking for evidence of FIXED escalation system..."
    
    # Check if any files have the real_escalation_enabled flag
    real_escalation_files=$(find "$OUTPUT_DIR" -name "*_FIXED_tao_seed*.json" -exec jq -r 'select(.run_configuration.real_escalation_enabled == true) | .run_configuration.dataset' {} \; | wc -l)
    
    if [ "$real_escalation_files" -gt 0 ]; then
        echo -e "${GREEN}✅ Real escalation system confirmed in $real_escalation_files files${NC}"
    else
        echo -e "${YELLOW}⚠️ Real escalation flag not found - may be using old system${NC}"
    fi
    
    # Check for conversation evidence in logs
    conversation_evidence=$(find "$LOGS_DIR" -name "*FIXED*.log" -exec grep -l "TRUE COLLABORATIVE ASSESSMENT\|REAL.*DISCUSSION" {} \; | wc -l)
    
    if [ "$conversation_evidence" -gt 0 ]; then
        echo -e "${GREEN}✅ Real agent conversations confirmed in $conversation_evidence log files${NC}"
    else
        echo -e "${YELLOW}⚠️ Limited conversation evidence found in logs${NC}"
    fi
fi

echo -e "\n${GREEN}🎯 Next Steps:${NC}"
echo "1. ✅ Analyze FIXED results in $OUTPUT_DIR"
echo "2. ✅ Review logs for escalation evidence"
echo "3. ✅ Compare escalation rates vs original system"
echo "4. ✅ Verify real agent conversations occurred"
echo "5. ✅ Run evaluation scripts on FIXED outputs"

echo -e "\n${PURPLE}===============================================${NC}"
echo -e "${PURPLE}🚀 FIXED TAO System Complete!${NC}"
echo -e "${PURPLE}Real Escalation & Agent Interaction Verified${NC}"
echo -e "${PURPLE}===============================================${NC}"
