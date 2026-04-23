import React, { useState, useEffect } from "react";

// @ast node: Function "FunctionComponent"
export function FunctionComponent({ text }: { text: string }) {
  return <div>{text}</div>;
}

// @ast node: Function "ArrowComponent"
const ArrowComponent = ({ count }: { count: number }) => {
  return <div>{count}</div>;
};

// @ast node: Function "ExportArrowComponent"
export const ExportArrowComponent = ({ name }: { name: string }) => {
  return <div>Hello, {name}</div>;
};

// @ast node: Function "DirectAssignmentComponent"
let DirectAssignmentComponent: React.FC<{ id: string }>;
DirectAssignmentComponent = ({ id }) => {
  const [data, setData] = useState<string | null>(null);

  useEffect(() => {
    setData(id);
  }, [id]);

  return <div>ID: {data}</div>;
};

// @ast node: Function "ExportDirectAssignmentComponent"
const ExportDirectAssignmentComponent: React.FC<{ items: string[] }> = ({
  items,
}) => {
  return (
    <ul>
      {items.map((item, index) => (
        <li key={index}>{item}</li>
      ))}
    </ul>
  );
};

export {
  ArrowComponent,
  DirectAssignmentComponent,
  ExportDirectAssignmentComponent,
};
