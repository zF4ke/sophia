import { AsyncLocalStorage } from "async_hooks";

type RequestCacheContextValue = {
    guildId: string | null;
    responseOrdinal: number | null;
};

const requestCacheContext = new AsyncLocalStorage<RequestCacheContextValue>();

export function runWithRequestCacheContext<T>(
    value: RequestCacheContextValue,
    callback: () => Promise<T>
): Promise<T> {
    return requestCacheContext.run(value, callback);
}

export function getRequestCacheContext(): RequestCacheContextValue {
    return requestCacheContext.getStore() ?? { guildId: null, responseOrdinal: null };
}
