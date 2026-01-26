/**
 * Format a date object to a readable string.
 * @param {Date} date
 * @returns {string}
 */
export function formatDate(date) {
  return new Intl.DateTimeFormat("en-US", {
    year: "numeric",
    month: "long",
    day: "numeric",
  }).format(date);
}

/**
 * Debounce a function call.
 * @param {Function} func
 * @param {number} wait
 */
export function debounce(func, wait) {
  let timeout;
  return function formatted(...args) {
    const context = this;
    clearTimeout(timeout);
    timeout = setTimeout(() => func.apply(context, args), wait);
  };
}
