import { defaultExclude, defineConfig } from 'vitest/config';

export default defineConfig({
	test: {
		globals: true,
		exclude: defaultExclude.concat(['tmp/**', 'dist/**', 'out/**']),
	},
});
