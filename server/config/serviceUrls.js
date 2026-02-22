const normalizeBaseUrl = raw => (raw || "").toString().trim().replace(/\/+$/, "");

const DOC_SERVICE_BASE_URL_FALLBACK = "http://localhost:8000";
export const DOC_SERVICE_BASE_URL = normalizeBaseUrl(
  process.env.DOC_SERVICE_BASE_URL || DOC_SERVICE_BASE_URL_FALLBACK
) || DOC_SERVICE_BASE_URL_FALLBACK;

export const docServiceUrl = path => {
  const safePath = (path || "").toString();
  return `${DOC_SERVICE_BASE_URL}${safePath.startsWith("/") ? safePath : `/${safePath}`}`;
};
