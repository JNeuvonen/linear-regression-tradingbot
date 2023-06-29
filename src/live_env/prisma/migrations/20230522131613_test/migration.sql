/*
  Warnings:

  - You are about to drop the column `position_size` on the `TradingStrategy` table. All the data in the column will be lost.

*/
-- AlterTable
ALTER TABLE "TradingStrategy" DROP COLUMN "position_size",
ADD COLUMN     "current_bet" DOUBLE PRECISION,
ADD COLUMN     "profit" DOUBLE PRECISION;
