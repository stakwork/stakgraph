// sherpa-onnx-node ships JSDoc typedefs, not .d.ts. The engine surface vein
// uses is typed locally in stt.ts (`SttEngine`); this only makes the lazy
// `import("sherpa-onnx-node")` compile.
declare module "sherpa-onnx-node" {
  const sherpa: any;
  export = sherpa;
}
