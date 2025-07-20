import { call } from '../src/tools/stagehand/tools.js';

async function testNetworkActivityGrouping() {
  console.log('Testing Network Activity Tool with Grouping...');
  
  const sessionId = 'test-grouping-session';
  
  try {
    // Test with Sphinx leaderboard (lots of repetitive requests)
    console.log('1. Navigating to Sphinx leaderboard (should generate many duplicate requests)...');
    const navResult = await call('stagehand_navigate', { url: 'https://community.sphinx.chat/leaderboard' }, sessionId);
    console.log('Navigation result:', navResult.isError ? 'ERROR' : 'SUCCESS');
    
    // Wait for all network activity to complete
    console.log('Waiting for network activity to complete...');
    await new Promise(resolve => setTimeout(resolve, 8000));
    
    // Test 2: Get network activity (simple mode with grouping)
    console.log('2. Getting network activity (simple mode with grouping)...');
    const networkResult = await call('stagehand_network_activity', {}, sessionId);
    
    if (networkResult.isError) {
      console.error('Network activity error:', networkResult.content[0].text);
      return;
    }
    
    const networkData = JSON.parse(networkResult.content[0].text);
    console.log('\n=== GROUPING RESULTS ===');
    console.log('Total entries found:', networkData.entries.length);
    console.log('Summary:', networkData.summary);
    
    // Count grouped vs individual entries
    const groupedEntries = networkData.entries.filter(entry => entry.grouped);
    const individualEntries = networkData.entries.filter(entry => !entry.grouped);
    
    console.log('\n=== BREAKDOWN ===');
    console.log('Grouped patterns:', groupedEntries.length);
    console.log('Individual entries:', individualEntries.length);
    
    // Show grouped entries
    if (groupedEntries.length > 0) {
      console.log('\n=== GROUPED PATTERNS ===');
      groupedEntries.forEach(group => {
        console.log(`${group.pattern}: ${group.count} requests (${group.successful} successful, ${group.failed} failed)`);
        console.log(`  Sample URLs: ${group.sample_urls.slice(0, 2).join(', ')}...`);
      });
    }
    
    // Show individual entries
    if (individualEntries.length > 0) {
      console.log('\n=== INDIVIDUAL ENTRIES ===');
      individualEntries.slice(0, 5).forEach(entry => {
        console.log(`${entry.method} ${entry.url} (${entry.type})`);
      });
      if (individualEntries.length > 5) {
        console.log(`... and ${individualEntries.length - 5} more individual entries`);
      }
    }
    
    // Test 3: Get network activity (verbose mode - should show all details)
    console.log('\n3. Testing verbose mode...');
    const verboseResult = await call('stagehand_network_activity', { verbose: true }, sessionId);
    
    if (!verboseResult.isError) {
      const verboseData = JSON.parse(verboseResult.content[0].text);
      console.log('Verbose mode entries found:', verboseData.length);
      
      // Count how many are grouped vs individual in verbose mode
      const verboseGrouped = verboseData.filter(entry => entry.grouped);
      const verboseIndividual = verboseData.filter(entry => !entry.grouped);
      console.log('Verbose grouped:', verboseGrouped.length);
      console.log('Verbose individual:', verboseIndividual.length);
    }
    
    console.log('\n✅ Network Activity Grouping test completed successfully!');
    
  } catch (error) {
    console.error('❌ Test failed:', error);
  }
}

// Run the test
testNetworkActivityGrouping().catch(console.error);