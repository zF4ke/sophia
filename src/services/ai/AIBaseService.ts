export class AIBaseService {
    protected static handleError(error: unknown, context: string): string {
        console.error(`AI Error in ${context}:`, error);
        return "Desculpe, ocorreu um erro ao processar sua solicitação.";
    }
}
