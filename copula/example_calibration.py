from calibration import ModelCalibrator


def main():
    """
    Example of using the model calibrator for a given ticker
    """
    # Ask user for ticker symbol
    ticker = input("Enter the stock symbol (e.g., AAPL, MSFT, GOOGL): ").upper()

    # Initialize calibrator
    print(f"Calibrating model for {ticker}...")
    calibrator = ModelCalibrator(ticker)

    # Fetch market data
    print("Fetching market data...")
    calibrator.fetch_market_data()

    # Check available expiry dates
    if not calibrator.expiry_dates or len(calibrator.expiry_dates) == 0:
        print("No expiry dates available for this ticker.")
        return

    # Display available expiry dates
    print("\nAvailable expiry dates:")
    for i, date in enumerate(calibrator.expiry_dates):
        print(f"{i+1}. {date}")

    # Select expiry date
    while True:
        try:
            choice = int(input("\nSelect an expiry date (number): "))
            if 1 <= choice <= len(calibrator.expiry_dates):
                selected_expiry = calibrator.expiry_dates[choice - 1]
                break
            else:
                print("Invalid choice. Please try again.")
        except ValueError:
            print("Please enter a number.")

    # Select option type
    while True:
        option_type = input("\nOption type (call/put): ").lower()
        if option_type in ["call", "put"]:
            break
        else:
            print("Invalid option type. Please enter 'call' or 'put'.")

    # Plot volatility smile
    print(f"\nPlotting volatility smile for {ticker} - {selected_expiry} ({option_type})...")
    calibrator.plot_volatility_smile(selected_expiry, option_type)

    # Calibrate model
    print(f"\nCalibrating model for {ticker} - {selected_expiry} ({option_type})...")
    params = calibrator.calibrate_model(selected_expiry, option_type)

    if params:
        print("\nCalibrated parameters:")
        print(f"Spot price (S0): {params['S0']:.2f}")
        print(f"Volatility (sigma): {params['sigma']:.2%}")
        print(f"Risk-free rate (r): {params['r']:.2%}")
        print(f"Dividend rate (q): {params['q']:.2%}")
        print(f"Time to expiry (T): {params['T']:.4f} years")
    else:
        print(
            "\nCalibration failed. Try with a different expiry date or option type."
        )


if __name__ == "__main__":
    main()
