import { h, render } from 'https://esm.sh/preact';
import { useState } from 'https://esm.sh/preact/hooks';
import htm from 'https://esm.sh/htm';
import { Prompt } from './prompt.js';

// Initialize HTM with Preact
const html = htm.bind(h);

// App Component - Parent of Prompt
const App = () => {
    const handleSend = (message) => {
        console.log('Message sent:', message);
        
        // Log the message text and tagged words
        console.log('Message text:', message.text);
        console.log('Tagged words:', message.taggedWords);
    };

    return html`
        <div>
            <${Prompt} onSend=${handleSend} />
        </div>
    `;
};

// Initialize the app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    render(h(App, null), document.getElementById('app'));
});