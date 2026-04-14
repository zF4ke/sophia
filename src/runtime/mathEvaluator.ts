/**
 * Safe recursive-descent math expression evaluator.
 * No eval/Function — parses and evaluates expressions directly.
 *
 * Supports:
 *   Operators: + - * / ^ (power)
 *   Functions: sqrt, abs, sin, cos, tan, log (base 10), ln, ceil, floor, round, min, max
 *   Constants: pi, e
 *   Parentheses, unary minus, implicit multiplication before functions/parens
 */

type Token =
    | { type: "number"; value: number }
    | { type: "op"; value: string }
    | { type: "func"; value: string }
    | { type: "lparen" }
    | { type: "rparen" }
    | { type: "comma" };

const FUNCTIONS = new Set([
    "sqrt", "abs", "sin", "cos", "tan",
    "log", "ln", "ceil", "floor", "round",
    "min", "max",
]);

const CONSTANTS: Record<string, number> = {
    pi: Math.PI,
    e: Math.E,
};

const MAX_EXPRESSION_LENGTH = 1000;

function tokenize(expr: string): Token[] {
    const tokens: Token[] = [];
    let i = 0;
    const s = expr.replace(/\s+/g, "");

    while (i < s.length) {
        const ch = s[i];

        if (/[0-9.]/.test(ch)) {
            let num = "";
            while (i < s.length && /[0-9.]/.test(s[i])) {
                num += s[i++];
            }
            const value = parseFloat(num);
            if (isNaN(value)) throw new Error(`Invalid number: ${num}`);
            tokens.push({ type: "number", value });
            continue;
        }

        if (/[a-zA-Z_]/.test(ch)) {
            let name = "";
            while (i < s.length && /[a-zA-Z_0-9]/.test(s[i])) {
                name += s[i++];
            }
            const lower = name.toLowerCase();
            if (CONSTANTS[lower] !== undefined) {
                tokens.push({ type: "number", value: CONSTANTS[lower] });
            } else if (FUNCTIONS.has(lower)) {
                tokens.push({ type: "func", value: lower });
            } else {
                throw new Error(`Unknown identifier: ${name}`);
            }
            continue;
        }

        if ("+-*/^".includes(ch)) {
            tokens.push({ type: "op", value: ch });
            i++;
            continue;
        }

        if (ch === "(") { tokens.push({ type: "lparen" }); i++; continue; }
        if (ch === ")") { tokens.push({ type: "rparen" }); i++; continue; }
        if (ch === ",") { tokens.push({ type: "comma" }); i++; continue; }

        throw new Error(`Unexpected character: ${ch}`);
    }

    return tokens;
}

class Parser {
    private pos = 0;
    constructor(private tokens: Token[]) {}

    private peek(): Token | null {
        return this.pos < this.tokens.length ? this.tokens[this.pos] : null;
    }

    private consume(): Token {
        if (this.pos >= this.tokens.length) throw new Error("Unexpected end of expression");
        return this.tokens[this.pos++];
    }

    private expect(type: string): Token {
        const t = this.consume();
        if (t.type !== type) throw new Error(`Expected ${type}, got ${t.type}`);
        return t;
    }

    private peekOp(): string | null {
        const t = this.peek();
        return t?.type === "op" ? t.value : null;
    }

    parse(): number {
        const result = this.parseAddSub();
        if (this.pos < this.tokens.length) {
            throw new Error(`Unexpected token at position ${this.pos}`);
        }
        return result;
    }

    private parseAddSub(): number {
        let left = this.parseMulDiv();
        let op = this.peekOp();
        while (op === "+" || op === "-") {
            this.consume();
            const right = this.parseMulDiv();
            left = op === "+" ? left + right : left - right;
            op = this.peekOp();
        }
        return left;
    }

    private parseMulDiv(): number {
        let left = this.parseUnary();
        let op = this.peekOp();
        while (op === "*" || op === "/") {
            this.consume();
            const right = this.parseUnary();
            if (op === "/" && right === 0) throw new Error("Division by zero");
            left = op === "*" ? left * right : left / right;
            op = this.peekOp();
        }
        return left;
    }

    private parseUnary(): number {
        const op = this.peekOp();
        if (op === "-") {
            this.consume();
            return -this.parsePower();
        }
        if (op === "+") {
            this.consume();
        }
        return this.parsePower();
    }

    private parsePower(): number {
        const base = this.parseAtom();
        if (this.peekOp() === "^") {
            this.consume();
            const exp = this.parseUnary(); // right-associative
            return Math.pow(base, exp);
        }
        return base;
    }

    private parseAtom(): number {
        const t = this.peek();
        if (!t) throw new Error("Unexpected end of expression");

        if (t.type === "number") {
            this.consume();
            return (t as { type: "number"; value: number }).value;
        }

        if (t.type === "func") {
            const name = (this.consume() as { type: "func"; value: string }).value;
            this.expect("lparen");
            const args: number[] = [this.parseAddSub()];
            while (this.peek()?.type === "comma") {
                this.consume();
                args.push(this.parseAddSub());
            }
            this.expect("rparen");
            return this.callFunction(name, args);
        }

        if (t.type === "lparen") {
            this.consume();
            const value = this.parseAddSub();
            this.expect("rparen");
            return value;
        }

        throw new Error(`Unexpected token: ${t.type}`);
    }

    private callFunction(name: string, args: number[]): number {
        switch (name) {
            case "sqrt":
                if (args.length !== 1) throw new Error("sqrt takes 1 argument");
                if (args[0] < 0) throw new Error("sqrt of negative number");
                return Math.sqrt(args[0]);
            case "abs":
                if (args.length !== 1) throw new Error("abs takes 1 argument");
                return Math.abs(args[0]);
            case "sin":
                if (args.length !== 1) throw new Error("sin takes 1 argument");
                return Math.sin(args[0]);
            case "cos":
                if (args.length !== 1) throw new Error("cos takes 1 argument");
                return Math.cos(args[0]);
            case "tan":
                if (args.length !== 1) throw new Error("tan takes 1 argument");
                return Math.tan(args[0]);
            case "log":
                if (args.length !== 1) throw new Error("log takes 1 argument");
                if (args[0] <= 0) throw new Error("log of non-positive number");
                return Math.log10(args[0]);
            case "ln":
                if (args.length !== 1) throw new Error("ln takes 1 argument");
                if (args[0] <= 0) throw new Error("ln of non-positive number");
                return Math.log(args[0]);
            case "ceil":
                if (args.length !== 1) throw new Error("ceil takes 1 argument");
                return Math.ceil(args[0]);
            case "floor":
                if (args.length !== 1) throw new Error("floor takes 1 argument");
                return Math.floor(args[0]);
            case "round":
                if (args.length !== 1) throw new Error("round takes 1 argument");
                return Math.round(args[0]);
            case "min":
                if (args.length < 2) throw new Error("min takes at least 2 arguments");
                return Math.min(...args);
            case "max":
                if (args.length < 2) throw new Error("max takes at least 2 arguments");
                return Math.max(...args);
            default:
                throw new Error(`Unknown function: ${name}`);
        }
    }
}

export function evaluateMath(expression: string): { result: number; expression: string } {
    if (expression.length > MAX_EXPRESSION_LENGTH) {
        throw new Error(`Expression too long (max ${MAX_EXPRESSION_LENGTH} characters)`);
    }
    const tokens = tokenize(expression);
    if (tokens.length === 0) throw new Error("Empty expression");
    const parser = new Parser(tokens);
    const result = parser.parse();
    if (!isFinite(result)) throw new Error("Result is not a finite number");
    return { result, expression };
}
