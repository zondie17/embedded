package decaf.driver.error;

import decaf.frontend.tree.Pos;

public class AbstractOverideError extends DecafError{

    private String clazz;

    public AbstractOverideError(Pos pos, String method){
        super(pos);
        this.clazz = method;
    }

    @Override
    protected String getErrMsg() {
        return "'" + clazz + "' is not abstract and does not override all abstract methods";
    }
}
