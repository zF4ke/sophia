import { z } from "zod";
import { T } from "@/shared/discordTools";
import { evaluateMath } from "@/runtime/mathEvaluator";
import type { ToolDefinition } from "./types";

const parameters = {
    type: "object",
    properties: {
        expression: {
            type: "string",
            description:
                "The mathematical expression to evaluate. Example: sqrt((328/3+23)^2 - 6)/6",
        },
    },
    required: ["expression"],
} as const;

export const evaluateMathTool: ToolDefinition = {
    name: T.evaluate_math,

    catalog: {
        effect: "read",
        description:
            "Evaluate a mathematical expression safely. Supports +, -, *, /, ^, sqrt, abs, sin, cos, tan, log, ln, pi, e.",
        evidenceRole: "discovery_only",
    },

    schema: {
        description:
            "Evaluate a mathematical expression and return the numeric result. Supports: +, -, *, /, ^ (power), sqrt(), abs(), sin(), cos(), tan(), log() (base 10), ln() (natural), ceil(), floor(), round(), min(), max(), constants pi and e. Example: sqrt((328/3+23)^2 - 6)/6",
        parameters,
    },

    capability: {
        description: "Evaluate a mathematical expression safely.",
        inputSchema: z.object({
            expression: z.string().describe("The math expression to evaluate."),
        }),
        outputSchema: z.any(),
        sideEffectLevel: "none",
        authRequirements: [],
        costClass: "cheap",
        latencyClass: "fast",
        preconditions: [],
        postconditions: ["returns the numeric result of the expression"],
        async run(_context, args) {
            const expression = String(args.expression || "");
            try {
                const { result } = evaluateMath(expression);
                return {
                    tool: T.evaluate_math,
                    summary: `${expression} = ${result}`,
                    data: { expression, result },
                };
            } catch (error) {
                const msg = error instanceof Error ? error.message : "Evaluation failed";
                return {
                    tool: T.evaluate_math,
                    summary: `Error: ${msg}`,
                    data: null,
                    errorMessage: msg,
                };
            }
        },
    },

    strategy: {
        extractEvidence() {
            return [];
        },
    },

    display: { icon: "🧮", labelPt: "Calcular expressão" },
};
