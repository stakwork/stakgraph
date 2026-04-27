// @ast node: DataModel "sveltekit"
// @ast node: DataModel "defineConfig"
// @ast node: DataModel "plugins: [sveltekit()]"
import { sveltekit } from '@sveltejs/kit/vite';
import { defineConfig } from 'vite';

export default defineConfig({
	plugins: [sveltekit()]
});
