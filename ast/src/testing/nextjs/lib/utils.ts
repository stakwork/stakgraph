import { clsx, type ClassValue } from "clsx"
import { twMerge } from "tailwind-merge"

// @ast node: Function "cn"
// @ast edge: Contains <- File "utils.ts" "src/testing/nextjs/lib/utils.ts"

// Merge class names conditionally
// Accepts any number of class values
export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}
