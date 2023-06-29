-- CreateTable
CREATE TABLE "TradingStrategy" (
    "id" SERIAL NOT NULL,
    "strategy_id" TEXT NOT NULL,
    "position_type" TEXT,
    "position_size" DOUBLE PRECISION NOT NULL,
    "strategy_init_balance" DOUBLE PRECISION NOT NULL,
    "strategy_curr_balance" DOUBLE PRECISION NOT NULL,
    "latest_log_message" TEXT NOT NULL,
    "latest_prediction" DOUBLE PRECISION NOT NULL,
    "updated_at" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "TradingStrategy_pkey" PRIMARY KEY ("id")
);

-- CreateIndex
CREATE UNIQUE INDEX "TradingStrategy_strategy_id_key" ON "TradingStrategy"("strategy_id");
