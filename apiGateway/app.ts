import createError from 'http-errors';
import express, {Request, Response, NextFunction} from 'express';
import cookieParser from 'cookie-parser';
import translationRouter from 'routes/api';
import authRouter from 'routes/auth';
import { requestLogger, responseLogger } from 'middleware/logging';

const app = express();

app.use(express.json());
app.use(express.urlencoded({ extended: false }));
app.use(cookieParser());
app.use(requestLogger);
app.use(responseLogger);

// enpoints/routes
app.use('/api', translationRouter);
app.use('/api/auth', authRouter);

// catch 404 and forward to error handler
app.use((_req: Request, _res: Response, next: NextFunction) => {
    next(createError(404));
});

// error handler
app.use((err: any, req: Request, res: Response, _next: NextFunction) => {
    const status = err.status || 500;
    const message = err.message || 'Internal Server Error';

    res.status(status).json({
        success: false,
        status,
        message,
        ...(req.app.get('env') === 'development' &&  { stack: err.stack })
    });
});

export default app;
