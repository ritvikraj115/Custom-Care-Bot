export function getErrorMessage(error, fallback) {
  if (error.response) {
    return error.response.data?.message || fallback;
  }
  return fallback;
}
