function rate = calculate_extendRate(rho, weight)
    if rho == 0
        rate = 0;
    else
        if(weight*rho > 1)
            rate = 1 + 1/(weight*rho);
        else
            rate = 1/(weight*rho);
        end
    end
end

