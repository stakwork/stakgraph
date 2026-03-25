import ReactMarkdown, { Components } from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkBreaks from "remark-breaks";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { tomorrow } from "react-syntax-highlighter/dist/esm/styles/prism";
import { cn } from "@/lib/utils";

interface MarkdownRendererProps {
  children: string;
  className?: string;
}

const styles = {
  text: "text-foreground",
  muted: "text-muted-foreground",
  border: "border-border",
  bg: "bg-muted/50",
  link: "text-primary",
  borderAccent: "border-primary",
};

const styleConfig = {
  heading: "font-semibold scroll-m-20",
  h1: "text-3xl lg:text-4xl mt-8 mb-4 border-b pb-2",
  h2: "text-2xl lg:text-3xl mt-6 mb-3",
  h3: "text-xl lg:text-2xl mt-5 mb-2",
  h4: "text-lg lg:text-xl mt-4 mb-2",
  h5: "text-base lg:text-lg mt-3 mb-1",
  h6: "text-sm lg:text-base mt-2 mb-1",
  paragraph: "leading-7 [&:not(:first-child)]:mt-4",
  blockquote: "border-l-4 pl-4 py-2 my-4 rounded-r-md italic",
  list: "my-4 ml-6 space-y-1 [&>li]:mt-1",
  listDisc: "list-disc",
  listDecimal: "list-decimal",
  listItem: "leading-7",
  codeInline: "relative rounded-xs px-0.75 py-0.5 text-sm font-mono bg-zinc-600/70",
  table: "w-full border-collapse",
  tableWrapper: "my-6 w-full overflow-y-auto rounded-lg border",
  tableHeader: "border-b font-medium [&>tr]:border-b",
  tableBody: "[&>tr:last-child]:border-0",
  tableRow: "border-b transition-colors hover:bg-muted/50",
  tableCell: "px-4 py-2 text-left align-middle",
  tableHeaderCell: "px-4 py-3 text-left font-semibold",
  image: "max-w-full h-auto rounded-lg border my-4 shadow-sm",
  hr: "my-8 border-t",
  link: "underline underline-offset-4 hover:opacity-80 transition-colors",
} as const;

const components: Components = {
  h1: ({ children, ...props }) => (
    <h1 className={cn(styleConfig.heading, styleConfig.h1, styles.text, styles.border)} {...props}>
      {children}
    </h1>
  ),
  h2: ({ children, ...props }) => (
    <h2 className={cn(styleConfig.heading, styleConfig.h2, styles.text)} {...props}>
      {children}
    </h2>
  ),
  h3: ({ children, ...props }) => (
    <h3 className={cn(styleConfig.heading, styleConfig.h3, styles.text)} {...props}>
      {children}
    </h3>
  ),
  h4: ({ children, ...props }) => (
    <h4 className={cn(styleConfig.heading, styleConfig.h4, styles.text)} {...props}>
      {children}
    </h4>
  ),
  h5: ({ children, ...props }) => (
    <h5 className={cn(styleConfig.heading, styleConfig.h5, styles.text)} {...props}>
      {children}
    </h5>
  ),
  h6: ({ children, ...props }) => (
    <h6 className={cn(styleConfig.heading, styleConfig.h6, styles.muted)} {...props}>
      {children}
    </h6>
  ),
  p: ({ children, ...props }) => (
    <p className={cn(styleConfig.paragraph, styles.text)} {...props}>
      {children}
    </p>
  ),
  em: ({ children, ...props }) => (
    <em className={cn("italic", styles.text)} {...props}>
      {children}
    </em>
  ),
  strong: ({ children, ...props }) => (
    <strong className={cn("font-semibold", styles.text)} {...props}>
      {children}
    </strong>
  ),
  blockquote: ({ children, ...props }) => (
    <blockquote
      className={cn(styleConfig.blockquote, styles.borderAccent, styles.bg, styles.muted)}
      {...props}
    >
      {children}
    </blockquote>
  ),
  ul: ({ children, ...props }) => (
    <ul className={cn(styleConfig.list, styleConfig.listDisc)} {...props}>
      {children}
    </ul>
  ),
  ol: ({ children, ...props }) => (
    <ol className={cn(styleConfig.list, styleConfig.listDecimal)} {...props}>
      {children}
    </ol>
  ),
  li: ({ children, ...props }) => (
    <li className={cn(styleConfig.listItem, styles.text)} {...props}>
      {children}
    </li>
  ),
  table: ({ children, ...props }) => (
    <div className={cn(styleConfig.tableWrapper, styles.border)}>
      <table className={styleConfig.table} {...props}>
        {children}
      </table>
    </div>
  ),
  thead: ({ children, ...props }) => (
    <thead className={cn(styleConfig.tableHeader, styles.bg)} {...props}>
      {children}
    </thead>
  ),
  tbody: ({ children, ...props }) => (
    <tbody className={styleConfig.tableBody} {...props}>
      {children}
    </tbody>
  ),
  tr: ({ children, ...props }) => (
    <tr className={styleConfig.tableRow} {...props}>
      {children}
    </tr>
  ),
  th: ({ children, ...props }) => (
    <th className={cn(styleConfig.tableHeaderCell, styles.text, styles.border)} {...props}>
      {children}
    </th>
  ),
  td: ({ children, ...props }) => (
    <td className={cn(styleConfig.tableCell, styles.text, styles.border)} {...props}>
      {children}
    </td>
  ),
  a: ({ children, href, ...props }) => (
    <a
      className={cn(styleConfig.link, styles.link, "break-all")}
      href={href}
      target={href?.startsWith("http") ? "_blank" : undefined}
      rel={href?.startsWith("http") ? "noopener noreferrer" : undefined}
      {...props}
    >
      {children}
    </a>
  ),
  img: ({ src, alt, ...props }) => (
    <img
      className={cn(styleConfig.image, styles.border)}
      src={src ?? ""}
      alt={alt || "Image"}
      loading="lazy"
      {...props}
    />
  ),
  hr: ({ ...props }) => <hr className={cn(styleConfig.hr, styles.border)} {...props} />,
  code: ({ className, children }) => {
    const match = /language-(\w+)/.exec(className || "");

    if (!match) {
      return (
        <code className={cn(styleConfig.codeInline, className)}>
          {children}
        </code>
      );
    }

    return (
      <SyntaxHighlighter
        PreTag="pre"
        wrapLines={true}
        language={match[1]}
        style={tomorrow}
      >
        {String(children).replace(/\n$/, "")}
      </SyntaxHighlighter>
    );
  },
};

export function MarkdownRenderer({ children, className }: MarkdownRendererProps) {
  const processedContent =
    typeof children === "string"
      ? children
          .replace(/\\n/g, "\n")
          .replace(/\\t/g, "\t")
          .replace(/\\"/g, '"')
          .replace(/\\'/g, "'")
      : children;

  return (
    <div className={cn("prose prose-invert max-w-full break-words", className)}>
      <ReactMarkdown
        remarkPlugins={[remarkGfm, remarkBreaks]}
        components={components}
      >
        {processedContent}
      </ReactMarkdown>
    </div>
  );
}
