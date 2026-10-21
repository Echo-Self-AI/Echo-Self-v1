package server

import (
	"os"
	"testing"
	"time"
)

func TestNotifyShutdownReturnsSignalChannel(t *testing.T) {
	signals := NotifyShutdown()

	if signals == nil {
		t.Fatal("expected a non-nil signal channel")
	}

	if cap(signals) != 1 {
		t.Fatalf("expected a buffered signal channel with capacity 1, got %d", cap(signals))
	}
}

func TestSignalChannelAcceptsInterrupt(t *testing.T) {
	signals := make(chan os.Signal, 1)
	signals <- os.Interrupt

	select {
	case got := <-signals:
		if got != os.Interrupt {
			t.Fatalf("expected interrupt signal, got %v", got)
		}
	case <-time.After(time.Second):
		t.Fatal("timed out waiting for interrupt signal")
	}
}
