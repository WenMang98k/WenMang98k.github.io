import { defineCollection, z } from 'astro:content';
import { glob } from 'astro/loaders';
const posts = defineCollection({
  loader: glob({ pattern: '**/*.md', base: './src/content/posts' }),
  schema: z.object({ title: z.string(), description: z.string(), date: z.coerce.date(), tags: z.array(z.string()), draft: z.boolean().default(false), series: z.string().optional(), seriesOrder: z.number().optional(), seriesRole: z.enum(['overview','method','resource']).optional() })
});
export const collections = { posts };
