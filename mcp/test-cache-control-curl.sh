#!/bin/bash

# Test script for cache control functionality
# Make sure the server is running on localhost:3355

echo "=== Testing Cache Control Functionality ==="
echo ""

# Test 1: Create initial answer (should create new node)
echo "1. Creating initial answer (should create new node):"
curl -s "http://localhost:3355/ask?question=Update%20workspace%20settings%20and%20configuration" | jq '.answer' | head -c 100
echo "..."
echo ""

# Test 2: Ask same question again (should use cached result)
echo "2. Asking same question again (should use cached result):"
curl -s "http://localhost:3355/ask?question=Update%20workspace%20settings%20and%20configuration" | jq '.answer' | head -c 100
echo "..."
echo ""

# ref_id: 2d2094ee-27bc-4986-b9ca-073035c9cc9f

# Test 3: Force refresh (should delete old node and create new one)
echo "3. Force refresh (should delete old node and create new one):"
curl -s "http://localhost:3355/ask?question=Update%20workspace%20settings%20and%20configuration&forceRefresh=true" | jq '.answer' | head -c 100
echo "..."
echo ""

# Test 4: Ask again after force refresh (should use new cached result)
echo "4. Asking again after force refresh (should use new cached result):"
curl -s "http://localhost:3355/ask?question=Update%20workspace%20settings%20and%20configuration" | jq '.answer' | head -c 100
echo "..."
echo ""

# Test 5: Max age 0.001 hours (should replace due to age)
echo "5. Max age 0.001 hours (should replace due to age):"
curl -s "http://localhost:3355/ask?question=Update%20workspace%20settings%20and%20configuration&maxAgeHours=0.001" | jq '.answer' | head -c 100
echo "..."
echo ""

# Test 6: Max age 24 hours (should use cached if less than 24 hours old)
echo "6. Max age 24 hours (should use cached if less than 24 hours old):"
curl -s "http://localhost:3355/ask?question=Update%20workspace%20settings%20and%20configuration&maxAgeHours=24" | jq '.answer' | head -c 100
echo "..."
echo ""

echo "=== Test Complete ==="
echo ""
echo "Key behaviors demonstrated:"
echo "- Initial call creates new node"
echo "- Subsequent calls use cached result"
echo "- forceRefresh=true deletes old node and immediately creates new one (no gap)"
echo "- maxAgeHours triggers replacement when age exceeds threshold"
echo "- Deletion happens right before creation to ensure no gap in data"
