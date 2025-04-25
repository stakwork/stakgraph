import React from 'react';

function RegularComponent(props: { message: string }) {
  return <div>{props.message}</div>;
}

export const ArrowComponent = (props: { count: number }) => {
  return <div>Count: {props.count}</div>;
};

export var DirectAssignmentComponent = (props: { value: string }) => {
  return <div>Value: {props.value}</div>;
};

export function ExportedFunctionComponent(props: { name: string }) {
  return <div>Hello, {props.name}!</div>;
}

export const ExportedArrowComponent = (props: { items: string[] }) => {
  return (
    <ul>
      {props.items.map((item, index) => (
        <li key={index}>{item}</li>
      ))}
    </ul>
  );
};

export default RegularComponent; 