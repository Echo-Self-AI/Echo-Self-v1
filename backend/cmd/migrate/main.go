package main

import (
	"flag"
	"fmt"
	"log"
	"os"
	"strconv"

	"github.com/Echo-Self-AI/Echo-Self-v1/backend/internal/config"
	"github.com/Echo-Self-AI/Echo-Self-v1/backend/internal/database"
)

func printUsage() {
	fmt.Print(`EchoSelf Database Migration Tool

Usage:
  go run ./cmd/migrate [command] [arguments]

Commands:
  up                 Apply all unapplied migrations
  down               Roll back all migrations
  step <n>           Apply n migrations (positive for up, negative for down)
  version            Print current migration version and dirty status
  force <version>    Set migration version and clear dirty state
  baseline [version] Baseline existing database without re-executing initial migrations (default: 1)
`)
}

func main() {
	flag.Parse()
	args := flag.Args()

	if len(args) == 0 {
		printUsage()
		os.Exit(1)
	}

	command := args[0]

	cfg := config.Load()
	if cfg.DatabaseURL == "" {
		log.Fatal("DATABASE_URL environment variable is required")
	}

	db, err := database.Connect(cfg.DatabaseURL)
	if err != nil {
		log.Fatalf("database connection failed: %v", err)
	}

	sqlDB, err := db.DB()
	if err != nil {
		log.Fatalf("get database handle failed: %v", err)
	}
	defer sqlDB.Close()

	switch command {
	case "up":
		if err := database.Migrate(db); err != nil {
			log.Fatalf("migration up failed: %v", err)
		}
		version, dirty, err := database.GetMigrationVersion(db)
		if err != nil {
			log.Fatalf("failed to retrieve version: %v", err)
		}
		log.Printf("migrations applied successfully; current version: %d (dirty: %t)", version, dirty)

	case "down":
		if err := database.MigrateDown(db); err != nil {
			log.Fatalf("migration down failed: %v", err)
		}
		log.Println("all migrations rolled back successfully")

	case "step":
		if len(args) < 2 {
			log.Fatal("step requires a step count argument (e.g. step 1 or step -1)")
		}
		n, err := strconv.Atoi(args[1])
		if err != nil {
			log.Fatalf("invalid step count '%s': %v", args[1], err)
		}
		if err := database.MigrateStep(db, n); err != nil {
			log.Fatalf("migration step (%d) failed: %v", n, err)
		}
		version, dirty, _ := database.GetMigrationVersion(db)
		log.Printf("migrated %d steps; current version: %d (dirty: %t)", n, version, dirty)

	case "version":
		version, dirty, err := database.GetMigrationVersion(db)
		if err != nil {
			log.Fatalf("failed to retrieve migration version: %v", err)
		}
		log.Printf("current migration version: %d (dirty: %t)", version, dirty)

	case "force":
		if len(args) < 2 {
			log.Fatal("force requires a version argument (e.g. force 1)")
		}
		v, err := strconv.Atoi(args[1])
		if err != nil {
			log.Fatalf("invalid version '%s': %v", args[1], err)
		}
		if err := database.ForceMigrationVersion(db, v); err != nil {
			log.Fatalf("failed to force version: %v", err)
		}
		log.Printf("successfully forced migration version to %d", v)

	case "baseline":
		baselineVer := 1
		if len(args) >= 2 {
			var err error
			baselineVer, err = strconv.Atoi(args[1])
			if err != nil {
				log.Fatalf("invalid baseline version '%s': %v", args[1], err)
			}
		}
		if err := database.Baseline(db, baselineVer); err != nil {
			log.Fatalf("failed to baseline database: %v", err)
		}
		version, dirty, err := database.GetMigrationVersion(db)
		if err != nil {
			log.Fatalf("failed to retrieve version: %v", err)
		}
		log.Printf("database baselined successfully; current version: %d (dirty: %t)", version, dirty)

	default:
		log.Fatalf("unknown command: %s", command)
	}
}
