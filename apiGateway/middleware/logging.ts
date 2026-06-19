import { Request, Response, NextFunction, RequestHandler } from 'express';
import { Server as WebSocketServer } from 'ws';
import logger from 'services/logger';

interface LoggedRequest extends Request {
  requestId?: string;
  startTime?: number;
}

let requestCounter = 0;

function generateRequestId(): string {
  requestCounter++;
  return `req-${Date.now()}-${requestCounter}`;
}

export function requestLogger(req: LoggedRequest, _res: Response, next: NextFunction): void {
  req.requestId = generateRequestId();
  req.startTime = Date.now();

  logger.info(`→ ${req.method} ${req.path}`, {
    requestId: req.requestId,
    method: req.method,
    path: req.path,
    userId: (req as any).user?.userId,
  });

  next();
}

export function responseLogger(req: LoggedRequest, res: Response, next: NextFunction): void {
  res.on('finish', () => {
    const duration = Date.now() - (req.startTime || Date.now());
    const statusCode = res.statusCode;

    const level = statusCode >= 500 ? 'error' : statusCode >= 400 ? 'warn' : 'info';

    logger.log(level, `← ${req.method} ${req.path} ${statusCode}`, {
      requestId: req.requestId,
      method: req.method,
      path: req.path,
      statusCode,
      duration,
      userId: (req as any).user?.userId,
    });
  });

  next();
}

export function withLogging(handler: RequestHandler, name: string): RequestHandler {
  return async (req: Request, res: Response, next: NextFunction) => {
    const loggedReq = req as LoggedRequest;

    logger.info(`→ [${name}]`, {
      requestId: loggedReq.requestId,
      method: req.method,
      path: req.path,
      userId: (req as any).user?.userId,
    });

    try {
      await handler(req, res, next);
    } catch (err: any) {
      logger.error(`✗ [${name}]`, {
        requestId: loggedReq.requestId,
        error: err.message,
        stack: err.stack,
        method: req.method,
        path: req.path,
        userId: (req as any).user?.userId,
      });
      next(err);
    }
  };
}

export function setupWsLogging(wss: WebSocketServer): void {
  wss.on('connection', (ws, req) => {
    const urlParams = new URLSearchParams(req.url?.split('?')[1] || '');
    const clientUuid = urlParams.get('uuid') || 'unknown';

    logger.info('WS connected', { uuid: clientUuid });

    ws.on('close', (code, reason) => {
      const reasonStr = Buffer.isBuffer(reason) ? reason.toString() : String(reason);
      logger.info('WS disconnected', { uuid: clientUuid, code, reason: reasonStr });
    });

    ws.on('error', (err) => {
      logger.error('WS error', { uuid: clientUuid, error: err.message });
    });
  });
}
