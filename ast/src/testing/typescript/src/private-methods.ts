// Deep coverage for private (#-prefixed) class members.
// Every private-method FORM should be captured as a Function node whose name
// keeps the leading '#'. A private field with a non-function value must NOT
// become a Function. Calls through a `this` receiver (this.#x()) are a separate
// limitation and are intentionally not asserted as Calls edges here.

// @ast node: Class "TokenVault"
export class TokenVault {
  // non-function private field — must NOT be captured as a Function
  #secret = "seed";

  // @ast node: Function "unlock"
  unlock(challenge: string): boolean {
    return this.#derive(challenge) === this.#secret;
  }

  // private instance method
  // @ast node: Function "#derive"
  #derive(salt: string): string {
    return salt + this.#secret;
  }

  // async private method
  // @ast node: Function "#rotate"
  async #rotate(): Promise<void> {
    this.#secret = await Promise.resolve(this.#derive("next"));
  }

  // static private method
  // @ast node: Function "#hash"
  static #hash(input: string): string {
    return input.split("").reverse().join("");
  }

  // private arrow-function field
  // @ast node: Function "#onChange"
  #onChange = (value: string): void => {
    this.#secret = value;
  };
}

// @ast node: Class "RequestSigner"
export class RequestSigner {
  // @ast node: Function "sign"
  sign(payload: string): string {
    return this.#compute(payload);
  }

  // @ast node: Function "#compute"
  #compute(payload: string): string {
    return payload.length.toString();
  }
}
