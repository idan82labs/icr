import { handleAuth, refreshToken } from '../auth/handler';
import { AuthToken, User, ApiResponse } from '../types/shared';

/**
 * POST /api/auth
 * Authenticates a user with the provided token.
 */
export async function authenticateEndpoint(
  request: { body: { token: string } }
): Promise<ApiResponse<User>> {
  const token: AuthToken = {
    token: request.body.token,
    userId: extractUserIdFromToken(request.body.token),
    expiresAt: extractExpiryFromToken(request.body.token),
  };

  const user = await handleAuth(token);

  if (!user) {
    return {
      success: false,
      error: 'Authentication failed',
      data: null,
    };
  }

  return {
    success: true,
    data: user,
  };
}

/**
 * POST /api/auth/refresh
 * Refreshes an expired or expiring token.
 */
export async function refreshEndpoint(
  request: { body: { token: string } }
): Promise<ApiResponse<AuthToken>> {
  const token = parseToken(request.body.token);
  const newToken = await refreshToken(token);

  if (!newToken) {
    return {
      success: false,
      error: 'Token refresh failed',
      data: null,
    };
  }

  return {
    success: true,
    data: newToken,
  };
}

function extractUserIdFromToken(token: string): string {
  // JWT parsing logic
  return 'user-id';
}

function extractExpiryFromToken(token: string): number {
  // JWT parsing logic
  return Date.now() + 3600000;
}

function parseToken(tokenString: string): AuthToken {
  return {
    token: tokenString,
    userId: extractUserIdFromToken(tokenString),
    expiresAt: extractExpiryFromToken(tokenString),
  };
}
