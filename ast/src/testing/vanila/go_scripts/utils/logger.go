package utils

import (
	"fmt"
	"log"
	"os"
	"time"
)

type Logger struct {
	infoLog  *log.Logger
	errorLog *log.Logger
}

func NewLogger() *Logger {
	return &Logger{
		infoLog:  log.New(os.Stdout, "INFO\t", log.Ldate|log.Ltime),
		errorLog: log.New(os.Stderr, "ERROR\t", log.Ldate|log.Ltime|log.Lshortfile),
	}
}

func (l *Logger) Info(format string, v ...interface{}) {
	l.infoLog.Printf(format, v...)
}

func (l *Logger) Error(format string, v ...interface{}) {
	l.errorLog.Printf(format, v...)
}

func (l *Logger) LogRequest(method, path string, duration time.Duration, status int) {
	l.infoLog.Printf("%s %s %d %v", method, path, status, duration)
}

func (l *Logger) Fatal(v ...interface{}) {
	l.errorLog.Output(2, fmt.Sprint(v...))
	os.Exit(1)
}
