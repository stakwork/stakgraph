import { Results, Assertion, Config } from "./types";

export const getTimeStamp = (): number => Date.now();

export const isInputOrTextarea = (element: Element): boolean =>
  element.tagName === "INPUT" ||
  element.tagName === "TEXTAREA" ||
  (element as HTMLElement).isContentEditable;

export const getElementSelector = (element: Element): string => {
  if (!element || element.nodeType !== 1) return "";

  const dataset = (element as HTMLElement).dataset;
  if (dataset?.testid) return `[data-testid="${dataset.testid}"]`;

  const id = (element as HTMLElement).id;
  if (id) return `#${id}`;

  let selector = element.tagName.toLowerCase();

  const className = (element as HTMLElement).className;
  if (className) {
    const classes = Array.from(element.classList)
      .filter((cls) => cls !== "staktrak-selection-active")
      .join(".");
    if (classes) selector += `.${classes}`;
  }

  if (element.tagName === "INPUT") {
    const type = (element as HTMLInputElement).type;
    if (type) selector += `[type="${type}"]`;
  }

  return selector;
};

export const createClickPath = (e: Event): string => {
  const path: string[] = [];
  (e as any).composedPath().forEach((el: Element, i: number) => {
    const composedPath = (e as any).composedPath();
    if (i < composedPath.length - 2) {
      let node = el.localName;
      const dataset = (el as HTMLElement).dataset;

      if (dataset?.testid) {
        node += `[data-testid="${dataset.testid}"]`;
      } else {
        const className = (el as HTMLElement).className;
        if (className) {
          el.classList.forEach((cls) => {
            if (cls !== "staktrak-selection-active") node += `.${cls}`;
          });
        }
        const id = (el as HTMLElement).id;
        if (id) node += `#${id}`;
      }
      path.push(node);
    }
  });
  return path.reverse().join(">");
};

export const filterClickDetails = (
  clickDetails: Array<[number, number, string, number]>,
  assertions: Assertion[],
  config: Config
): Array<[number, number, string, number]> => {
  if (!clickDetails.length) return [];

  let filtered = config.filterAssertionClicks
    ? clickDetails.filter(
        (click) =>
          !assertions.some(
            (assertion) =>
              Math.abs(click[3] - assertion.timestamp) < 1000 &&
              (click[2].includes(assertion.selector) ||
                assertion.selector.includes(click[2]))
          )
      )
    : clickDetails;

  // Remove rapid multi-clicks
  const clicksBySelector: Record<
    string,
    Array<{ detail: any; timestamp: number }>
  > = {};
  filtered.forEach((click) => {
    const selector = click[2];
    if (!clicksBySelector[selector]) clicksBySelector[selector] = [];
    clicksBySelector[selector].push({ detail: click, timestamp: click[3] });
  });

  const result: Array<[number, number, string, number]> = [];
  Object.values(clicksBySelector).forEach((clicks) => {
    clicks.sort((a, b) => a.timestamp - b.timestamp);
    let lastClick: any = null;

    clicks.forEach((click) => {
      if (
        !lastClick ||
        click.timestamp - lastClick.timestamp > config.multiClickInterval
      ) {
        result.push(click.detail);
      }
      lastClick = click;
    });
  });

  return result.sort((a, b) => a[3] - b[3]);
};
