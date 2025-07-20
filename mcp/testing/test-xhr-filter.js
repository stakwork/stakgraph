import { call } from '../src/tools/stagehand/tools.js';

async function testSimplifiedNetworkTool() {
  console.log('Testing Simplified Network Tool (XHR/Fetch Only)...');
  
  const sessionId = 'test-simplified-session';
  
  try {
    // Navigate to Sphinx leaderboard
    console.log('1. Navigating to Sphinx leaderboard...');
    const navResult = await call('stagehand_navigate', { url: 'https://community.sphinx.chat/leaderboard' }, sessionId);
    console.log('Navigation result:', navResult.isError ? 'ERROR' : 'SUCCESS');
    
    // Wait for network activity
    console.log('Waiting for network activity...');
    await new Promise(resolve => setTimeout(resolve, 8000));
    
    // Test default behavior (all = xhr + fetch)
    console.log('2. Getting API calls (default: all = xhr + fetch)...');
    const allResult = await call('stagehand_network_activity', { filter: 'all' }, sessionId);
    
    if (allResult.isError) {
      console.error('All filtering error:', allResult.content[0].text);
      return;
    }
    
    const allData = JSON.parse(allResult.content[0].text);
    console.log('\n=== API CALLS RESULTS ===');
    console.log('Total API entries:', allData.entries.length);
    console.log('Summary:', allData.summary);
    
    // Count by resource type
    const resourceTypes = {};
    allData.entries.forEach(entry => {
      resourceTypes[entry.resourceType] = (resourceTypes[entry.resourceType] || 0) + 1;
    });
    
    console.log('\n=== API CALL BREAKDOWN ===');
    Object.entries(resourceTypes).forEach(([type, count]) => {
      console.log(`${type}: ${count} entries`);
    });
    
    // Show first few API calls
    console.log('\n=== SAMPLE API CALLS ===');
    allData.entries.slice(0, 10).forEach((entry, index) => {
      const status = entry.status ? `→ ${entry.status}` : '';
      const duration = entry.duration ? `(${entry.duration}ms)` : '';
      console.log(`${index + 1}. ${entry.method} ${entry.url} ${status} ${duration}`);
    });
    
    // Test XHR only
    console.log('\n3. Getting XHR requests only...');
    const xhrResult = await call('stagehand_network_activity', { filter: 'xhr' }, sessionId);
    
    if (!xhrResult.isError) {
      const xhrData = JSON.parse(xhrResult.content[0].text);
      console.log('XHR only entries:', xhrData.entries.length);
    }
    
    // Test fetch only
    console.log('\n4. Getting fetch requests only...');
    const fetchResult = await call('stagehand_network_activity', { filter: 'fetch' }, sessionId);
    
    if (!fetchResult.isError) {
      const fetchData = JSON.parse(fetchResult.content[0].text);
      console.log('Fetch only entries:', fetchData.entries.length);
      
      if (fetchData.entries.length > 0) {
        console.log('First fetch request:', fetchData.entries[0]);
      }
    }
    
    console.log('\n✅ Simplified Network Tool test completed!');
    console.log('🎯 Result: Clean API calls only, no resource noise!');
    
  } catch (error) {
    console.error('❌ Test failed:', error);
  }
}

// Run the test
testSimplifiedNetworkTool().catch(console.error);