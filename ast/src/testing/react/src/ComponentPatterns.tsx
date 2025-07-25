import React, { useState, useEffect } from "react";

export function FunctionComponent({ text }: { text: string }) {
  return <div>{text}</div>;
}

const ArrowComponent = ({ count }: { count: number }) => {
  return <div>{count}</div>;
};

export const ExportArrowComponent = ({ name }: { name: string }) => {
  return <div>Hello, {name}</div>;
};

let DirectAssignmentComponent: React.FC<{ id: string }>;
DirectAssignmentComponent = ({ id }) => {
  const [data, setData] = useState<string | null>(null);

  useEffect(() => {
    setData(id);
  }, [id]);

  return <div>ID: {data}</div>;
};

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
