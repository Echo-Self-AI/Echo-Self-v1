package server

import (
	"os"
	"os/signal"
	"syscall"
)

// NotifyShutdown returns a channel that receives interrupt and termination signals.
func NotifyShutdown() <-chan os.Signal {
	signals := make(chan os.Signal, 1)

	signal.Notify(signals, os.Interrupt, syscall.SIGTERM)

	return signals
}
